package task3;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.*;
import java.net.URI;
import java.util.*;

// =============================================================================
// APPROACH A – Map-function level local aggregation for Stripes
// =============================================================================
/**
 * Stripes Mapper with MAP-FUNCTION LEVEL local aggregation.
 *
 * A per-centre-word HashMap is built within each map() call (per line),
 * then emitted. This is effectively what the basic StripesMapper already
 * does, but here we make the intent explicit and extend it slightly:
 * the HashMap accumulates counts for MULTIPLE occurrences of the same
 * centre word within the same line.
 *
 * Difference from the basic Stripes Mapper in Task 2:
 *   Task 2: one HashMap per centre word per line iteration step.
 *   Here:   one HashMap PER UNIQUE CENTRE WORD across the entire line.
 *           e.g. if "data" appears 3 times in a line we emit ONE stripe
 *           for "data" with all neighbours merged, not 3 partial stripes.
 */
public class StripesMapFunctionLevelMapper
        extends Mapper<Object, Text, Text, Text> {

    private final Set<String> vocabulary = new HashSet<>();
    private final Text        wordKey    = new Text();
    private final Text        stripeVal  = new Text();

    private int windowSize = 2;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        windowSize = context.getConfiguration().getInt("stripes.window.size", 2);
        loadVocabulary(context);
    }

    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        // ---- Map-function level store: centre → (neighbour → count) ----
        Map<String, Map<String, Integer>> lineStripes = new HashMap<>();

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            Map<String, Integer> stripe =
                lineStripes.computeIfAbsent(center, k -> new HashMap<>());

            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {
                if (j == i) continue;
                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                stripe.merge(neighbour, 1, Integer::sum);
            }
        }

        // Flush: emit one stripe per unique centre word per line
        for (Map.Entry<String, Map<String, Integer>> e : lineStripes.entrySet()) {
            wordKey.set(e.getKey());
            stripeVal.set(serialise(e.getValue()));
            context.write(wordKey, stripeVal);
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

    private static String serialise(Map<String, Integer> stripe) {
        StringBuilder sb = new StringBuilder();
        stripe.forEach((k, v) -> {
            if (sb.length() > 0) sb.append('\t');
            sb.append(k).append(':').append(v);
        });
        return sb.toString();
    }
}


// =============================================================================
// APPROACH B – Map-class level local aggregation for Stripes (In-Mapper Combining)
// =============================================================================
/**
 * Stripes Mapper with MAP-CLASS LEVEL local aggregation (In-Mapper Combining).
 *
 * A two-level HashMap (centre → neighbour → count) is maintained across ALL
 * map() calls for the entire task/split.  cleanup() flushes it in one pass.
 *
 * Memory analysis:
 *   - Outer map: at most V=50 entries (one per vocabulary word).
 *   - Inner map: at most V-1=49 entries per centre word.
 *   - Total cells: ≤ 50×49 = 2450, each storing a small integer.
 *   - This is negligible — the In-Mapper Combining pattern is IDEAL for
 *     bounded vocabularies like our top-50 word set.
 *
 * Shuffle reduction:
 *   Without In-Mapper Combining: one stripe emitted per centre word per line.
 *   With In-Mapper Combining:    one stripe emitted per centre word per SPLIT.
 *   For a 128 MB split of Wikipedia text this can be an order-of-magnitude
 *   reduction in data crossing the network.
 *
 * Stripes vs Pairs (In-Mapper Combining comparison):
 * ┌─────────────────────┬──────────────────────┬──────────────────────┐
 * │ Metric              │ Pairs + IMC          │ Stripes + IMC        │
 * ├─────────────────────┼──────────────────────┼──────────────────────┤
 * │ Keys emitted        │ ≤ V² = 2500 / split  │ ≤ V = 50 / split     │
 * │ Values emitted      │ integer per pair     │ serialised map/word  │
 * │ Shuffle bytes       │ lower count, small V │ far fewer keys, but  │
 * │                     │                      │ values are larger    │
 * │ Reducer complexity  │ simple sum           │ map merge            │
 * │ Memory (mapper)     │ O(V²) ints           │ O(V²) ints (same)    │
 * └─────────────────────┴──────────────────────┴──────────────────────┘
 *
 * For V=50 both approaches are cheap. Stripes wins on key count; Pairs wins
 * on value simplicity.  For large V (e.g. full vocabulary), Stripes is
 * significantly more efficient due to the massive key reduction.
 */
class StripesClassLevelMapper extends Mapper<Object, Text, Text, Text> {

    private final Set<String> vocabulary = new HashSet<>();

    // ---- Class-level aggregation store ----
    // Outer key: centre word
    // Inner key: neighbour word
    // Value    : accumulated co-occurrence count
    private final Map<String, Map<String, Integer>> globalStripes = new HashMap<>();

    private       int  windowSize = 2;
    private final Text wordKey    = new Text();
    private final Text stripeVal  = new Text();

    // -----------------------------------------------------------------------
    // setup() – initialise once per task
    // -----------------------------------------------------------------------
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        windowSize = context.getConfiguration().getInt("stripes.window.size", 2);
        loadVocabulary(context);
        // Pre-populate outer map for all vocabulary words (optional optimisation
        // to avoid computeIfAbsent overhead in hot map() loop)
        for (String word : vocabulary) {
            globalStripes.put(word, new HashMap<>());
        }
    }

    // -----------------------------------------------------------------------
    // map() – update global store; NO emit
    // -----------------------------------------------------------------------
    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            Map<String, Integer> stripe = globalStripes.get(center);

            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {
                if (j == i) continue;
                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                stripe.merge(neighbour, 1, Integer::sum);
            }
        }
        // Intentionally no context.write() here
    }

    // -----------------------------------------------------------------------
    // cleanup() – single flush after all lines in the split are processed
    // -----------------------------------------------------------------------
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (Map.Entry<String, Map<String, Integer>> outer : globalStripes.entrySet()) {
            Map<String, Integer> stripe = outer.getValue();
            if (stripe.isEmpty()) continue;   // skip words not seen in this split

            wordKey.set(outer.getKey());
            stripeVal.set(serialise(stripe));
            context.write(wordKey, stripeVal);
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

    private static String serialise(Map<String, Integer> stripe) {
        StringBuilder sb = new StringBuilder();
        stripe.forEach((k, v) -> {
            if (sb.length() > 0) sb.append('\t');
            sb.append(k).append(':').append(v);
        });
        return sb.toString();
    }
}
