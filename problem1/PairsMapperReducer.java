package task2.pairs;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.*;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;

// =============================================================================
// Pairs Mapper
// =============================================================================
/**
 * Co-occurrence Mapper – Pairs Approach.
 *
 * For every word w_i in the top-50 vocabulary, it slides a window of radius d
 * over the sentence and emits one key-value pair per co-occurring neighbour:
 *
 *   key  : "(wordA, wordB)"   ← serialised as a single Text token
 *   value: 1
 *
 * Memory footprint: O(V·d) per line at most, dominated by the window buffer.
 *
 * Trade-off vs Stripes:
 *   + Each mapper call is cheap; no intermediate HashMap per word.
 *   – Many more key-value pairs cross the network (one per ordered pair).
 *   – Reducers must handle a very large number of unique keys.
 */
public class PairsMapper extends Mapper<Object, Text, Text, IntWritable> {

    private static final IntWritable ONE = new IntWritable(1);

    private final Set<String> vocabulary = new HashSet<>();
    private final Text pairKey = new Text();

    private int windowSize = 2;   // default window radius d = 2

    // -----------------------------------------------------------------------
    // setup – load top-50 vocabulary and window size from configuration
    // -----------------------------------------------------------------------
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        // Window size can be configured: -D pairs.window.size=<d>
        windowSize = context.getConfiguration().getInt("pairs.window.size", 2);

        // Load top-50 vocabulary from Distributed Cache
        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles != null && cacheFiles.length > 0) {
            Path vocabPath = new Path(cacheFiles[0].getPath());
            try (BufferedReader br = new BufferedReader(
                    new FileReader(vocabPath.getName()))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length > 0) vocabulary.add(parts[0].toLowerCase());
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // map – emit (w_i, w_j) for every in-vocabulary pair within window
    // -----------------------------------------------------------------------
    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        for (int i = 0; i < tokens.length; i++) {
            String center = tokens[i];
            if (!vocabulary.contains(center)) continue;

            // Slide window [i-d, i+d], excluding i itself
            for (int j = Math.max(0, i - windowSize);
                     j <= Math.min(tokens.length - 1, i + windowSize); j++) {

                if (j == i) continue;

                String neighbour = tokens[j];
                if (!vocabulary.contains(neighbour)) continue;

                // Emit ordered pair (center, neighbour)
                pairKey.set("(" + center + "," + neighbour + ")");
                context.write(pairKey, ONE);
            }
        }
    }
}


// =============================================================================
// Pairs Reducer
// =============================================================================
/**
 * Co-occurrence Reducer – Pairs Approach.
 *
 * Receives ("(wordA,wordB)", [1, 1, ...]) and emits the total count.
 * Simple summation – the complexity lives in the large number of unique keys.
 */
class PairsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    private final IntWritable result = new IntWritable();

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {

        int sum = 0;
        for (IntWritable v : values) sum += v.get();

        result.set(sum);
        context.write(key, result);
    }
}
