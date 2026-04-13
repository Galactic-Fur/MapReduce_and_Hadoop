package task2.stripes;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.*;
import java.net.URI;
import java.nio.file.Path;
import java.util.*;

// =============================================================================
// Custom MapWritable-like helper – using a simple serialised string map
// so the code compiles without extra Writable boilerplate.
// In production you would extend MapWritable or use a custom Writable.
// =============================================================================

// =============================================================================
// Stripes Mapper
// =============================================================================
/**
 * Co-occurrence Mapper – Stripes Approach.
 *
 * For every in-vocabulary word w_i the mapper builds a partial "stripe":
 * a HashMap of all co-occurring neighbours and their counts within window d.
 * The stripe is serialised to a Tab-separated string and emitted once per
 * centre word per line:
 *
 *   key  : "wordA"
 *   value: "wordB:3\twordC:1\t..."   ← partial stripe
 *
 * Trade-off vs Pairs:
 *   + Far fewer keys emitted (one per centre word, not one per pair).
 *   + Reducers aggregate entire stripes — fewer reduce calls.
 *   – Each map() call materialises a HashMap → higher per-call memory usage.
 *   – Serialisation/deserialisation overhead for the stripe value.
 *   – Reducer merge logic is more complex than simple integer summation.
 */
public class StripesMapper extends Mapper<Object, Text, Text, Text> {

    private final Set<String> vocabulary = new HashSet<>();
    private final Text wordKey   = new Text();
    private final Text stripeVal = new Text();

    private int windowSize = 2;

    // -----------------------------------------------------------------------
    // setup
    // -----------------------------------------------------------------------
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        windowSize = context.getConfiguration().getInt("stripes.window.size", 2);

        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles != null && cacheFiles.length > 0) {
            java.nio.file.Path vocabPath =
                java.nio.file.Paths.get(cacheFiles[0].getPath());
            try (BufferedReader br = new BufferedReader(
                    new FileReader(vocabPath.getFileName().toString()))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length > 0) vocabulary.add(parts[0].toLowerCase());
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // map – build a stripe per centre word per line, then emit it
    // -----------------------------------------------------------------------
    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            // Build partial stripe for this occurrence of 'center'
            Map<String, Integer> stripe = new HashMap<>();

            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {

                if (j == i) continue;
                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                stripe.merge(neighbour, 1, Integer::sum);
            }

            if (stripe.isEmpty()) continue;

            // Serialise stripe as "word1:count1\tword2:count2\t..."
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Integer> e : stripe.entrySet()) {
                if (sb.length() > 0) sb.append('\t');
                sb.append(e.getKey()).append(':').append(e.getValue());
            }

            wordKey.set(center);
            stripeVal.set(sb.toString());
            context.write(wordKey, stripeVal);
        }
    }
}


// =============================================================================
// Stripes Reducer
// =============================================================================
/**
 * Co-occurrence Reducer – Stripes Approach.
 *
 * Receives (wordA, [stripe1, stripe2, ...]) and merges all partial stripes
 * into a single aggregated stripe, then emits (wordA, mergedStripe).
 *
 * The merge is simply entry-by-entry integer addition across all partial maps.
 */
class StripesReducer extends Reducer<Text, Text, Text, Text> {

    private final Text result = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {

        Map<String, Integer> merged = new HashMap<>();

        for (Text val : values) {
            // Deserialise partial stripe
            for (String entry : val.toString().split("\t")) {
                String[] kv = entry.split(":");
                if (kv.length != 2) continue;
                String neighbour = kv[0];
                int    count;
                try { count = Integer.parseInt(kv[1]); }
                catch (NumberFormatException e) { continue; }

                merged.merge(neighbour, count, Integer::sum);
            }
        }

        // Serialise the merged stripe
        StringBuilder sb = new StringBuilder();
        // Sort for deterministic output
        new TreeMap<>(merged).forEach((k, v) -> {
            if (sb.length() > 0) sb.append('\t');
            sb.append(k).append(':').append(v);
        });

        result.set(sb.toString());
        context.write(key, result);
    }
}
