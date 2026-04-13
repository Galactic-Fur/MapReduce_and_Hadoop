package task1;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

// =============================================================================
// TopN Mapper
// =============================================================================
/**
 * Second-pass Mapper for extracting the Top-N words.
 *
 * Reads (word, count) pairs produced by WordCountReducer and maintains an
 * in-memory TreeMap of size N.  All output is flushed in cleanup() so only
 * N candidates per mapper split travel across the network.
 *
 * Key design: using NullWritable as the output key forces all records to the
 * SAME reducer, which then performs the final global top-N selection.
 */
public class TopNMapper extends Mapper<Object, Text, NullWritable, Text> {

    private static final int N = 50;

    // TreeMap keyed by count (ascending) – oldest minimum is easily removed
    private final TreeMap<Integer, String> topMap = new TreeMap<>();

    @Override
    protected void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {

        String[] parts = value.toString().trim().split("\\s+");
        if (parts.length != 2) return;

        String word  = parts[0];
        int    count;
        try {
            count = Integer.parseInt(parts[1]);
        } catch (NumberFormatException e) {
            return;
        }

        topMap.put(count, word);

        // Keep the map bounded to N entries
        if (topMap.size() > N) {
            topMap.pollFirstEntry();   // remove the smallest count
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        // Emit the local top-N as "word\tcount" strings
        for (Map.Entry<Integer, String> entry : topMap.entrySet()) {
            context.write(NullWritable.get(),
                          new Text(entry.getValue() + "\t" + entry.getKey()));
        }
    }
}


// =============================================================================
// TopN Reducer
// =============================================================================
/**
 * Second-pass Reducer for extracting the Top-N words.
 *
 * Because all mappers emit NullWritable keys, a single reducer receives every
 * candidate.  It rebuilds the TreeMap and emits the final top-N sorted
 * descending by frequency.
 */
class TopNReducer extends Reducer<NullWritable, Text, Text, IntWritable> {

    private static final int N = 50;

    private final TreeMap<Integer, String> topMap = new TreeMap<>();

    @Override
    protected void reduce(NullWritable key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {

        for (Text val : values) {
            String[] parts = val.toString().split("\t");
            if (parts.length != 2) continue;

            String word  = parts[0];
            int    count;
            try {
                count = Integer.parseInt(parts[1]);
            } catch (NumberFormatException e) {
                continue;
            }

            topMap.put(count, word);
            if (topMap.size() > N) {
                topMap.pollFirstEntry();
            }
        }

        // Emit top-N in descending order of frequency
        for (Map.Entry<Integer, String> entry : topMap.descendingMap().entrySet()) {
            context.write(new Text(entry.getValue()),
                          new IntWritable(entry.getKey()));
        }
    }
}
