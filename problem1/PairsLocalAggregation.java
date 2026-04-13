package task3;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.*;
import java.net.URI;
import java.util.*;

// =============================================================================
// APPROACH A – Map-function level aggregation (HashMap inside map())
// =============================================================================
/**
 * Pairs Mapper with MAP-FUNCTION LEVEL local aggregation.
 *
 * An associative array (HashMap) is allocated PER LINE inside map().
 * All (pair → count) accumulations for that line are flushed at the end
 * of the same map() call. This reduces duplicate pair emissions within a
 * single line but resets after every call — so a word appearing 10 times
 * in a line produces one emit, not 10.
 *
 * When to prefer this:
 *   - Lines are long and contain many repeated tokens.
 *   - You want simple, stateless map() calls without setup/cleanup.
 *   - Memory is very limited (HashMap is bounded by line length × window).
 *
 * Limitation:
 *   - No aggregation ACROSS lines in the same split.
 *   - The same pair (A, B) appearing in 1000 different lines still
 *     generates 1000 emits (one per line at minimum).
 */
public class PairsMapFunctionLevelMapper
        extends Mapper<Object, Text, Text, IntWritable> {

    private final Set<String> vocabulary = new HashSet<>();
    private final Text        pairKey    = new Text();

    private int windowSize = 2;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        windowSize = context.getConfiguration().getInt("pairs.window.size", 2);
        loadVocabulary(context);
    }

    /**
     * Local aggregation at the MAP-FUNCTION level.
     *
     * Key insight: instead of emitting (pair, 1) for every co-occurrence,
     * we tally within a local HashMap and emit (pair, localCount) once
     * per unique pair per line.
     */
    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        // ----- Local associative array for this single map() call -----
        Map<String, Integer> localCounts = new HashMap<>();

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {
                if (j == i) continue;
                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                String pair = "(" + center + "," + neighbour + ")";
                localCounts.merge(pair, 1, Integer::sum);
            }
        }

        // Flush local counts to context
        for (Map.Entry<String, Integer> entry : localCounts.entrySet()) {
            pairKey.set(entry.getKey());
            context.write(pairKey, new IntWritable(entry.getValue()));
        }
    }

    // -----------------------------------------------------------------------
    private void loadVocabulary(Context context) throws IOException {
        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles == null || cacheFiles.length == 0) return;

        String fileName = new org.apache.hadoop.fs.Path(
            cacheFiles[0].getPath()).getName();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length > 0) vocabulary.add(parts[0].toLowerCase());
            }
        }
    }
}


// =============================================================================
// APPROACH B – Map-class level aggregation (setup / cleanup pattern)
// =============================================================================
/**
 * Pairs Mapper with MAP-CLASS LEVEL local aggregation (In-Mapper Combining).
 *
 * A single HashMap is SHARED across ALL map() calls in this task attempt
 * (i.e., the entire split).  Intermediate counts accumulate in memory and
 * are only flushed in cleanup() — once per task.
 *
 *   setup()   → initialise shared HashMap
 *   map()     → update HashMap, NO context.write()
 *   cleanup() → iterate HashMap, emit (pair, totalLocalCount)
 *
 * This is the canonical "In-Mapper Combining" pattern from Lin & Dyer.
 *
 * Benefits:
 *   - Maximally reduces the number of key-value pairs emitted by the mapper.
 *   - Eliminates the overhead of the local Combiner (no serialise-deserialise
 *     cycle between map and combine phases on the same node).
 *   - Particularly effective when the vocabulary is small (fixed at 50 words)
 *     so the HashMap size is bounded by V² = 2500 entries at most.
 *
 * Trade-offs:
 *   - HashMap lives in JVM heap for the entire task duration.
 *     For very large splits this can cause OOM errors.
 *   - If a task is retried (speculative execution, node failure), the
 *     in-memory state is lost — but MapReduce restarts are idempotent so
 *     this is safe.
 *   - The Reducer still receives counts aggregated per split, which reduces
 *     shuffle traffic significantly vs the naive approach.
 *
 * When the vocabulary is bounded (V=50) this pattern is virtually free of
 * memory risk and is strongly recommended over the plain combiner approach.
 */
class PairsClassLevelMapper
        extends Mapper<Object, Text, Text, IntWritable> {

    private final Set<String>         vocabulary  = new HashSet<>();
    // ---- Class-level aggregation store (shared across all map() calls) ----
    private final Map<String, Integer> pairCounts  = new HashMap<>();

    private       int  windowSize = 2;
    private final Text pairKey    = new Text();

    // -----------------------------------------------------------------------
    // setup() – runs ONCE per Mapper task (before any map() call)
    // -----------------------------------------------------------------------
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        windowSize = context.getConfiguration().getInt("pairs.window.size", 2);
        loadVocabulary(context);
        // pairCounts is already initialised as an empty HashMap
    }

    // -----------------------------------------------------------------------
    // map() – accumulate into the class-level HashMap; DO NOT write to context
    // -----------------------------------------------------------------------
    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {
                if (j == i) continue;
                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                // Accumulate — no emit here
                String pair = "(" + center + "," + neighbour + ")";
                pairCounts.merge(pair, 1, Integer::sum);
            }
        }
        // Notice: context.write() is intentionally absent
    }

    // -----------------------------------------------------------------------
    // cleanup() – runs ONCE after ALL map() calls complete for this task.
    //             This is the single flush point.
    // -----------------------------------------------------------------------
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
            pairKey.set(entry.getKey());
            context.write(pairKey, new IntWritable(entry.getValue()));
        }
    }

    // -----------------------------------------------------------------------
    private void loadVocabulary(Context context) throws IOException {
        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles == null || cacheFiles.length == 0) return;

        String fileName = new org.apache.hadoop.fs.Path(
            cacheFiles[0].getPath()).getName();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length > 0) vocabulary.add(parts[0].toLowerCase());
            }
        }
    }
}
