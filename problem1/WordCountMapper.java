package task1;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.*;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;

/**
 * Task 1 – Word Count Mapper (Pairs approach, stop-word filtering).
 *
 * "Pairs approach" here means each (word, 1) emitted by the mapper is
 * treated as the elementary unit / pair, and the Reducer sums the counts.
 * The stop-words file is shipped to every TaskTracker node via the
 * Hadoop Distributed Cache so that the Mapper can filter locally without
 * any extra network traffic.
 */
public class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------
    private static final IntWritable ONE = new IntWritable(1);

    // ---------------------------------------------------------------
    // Instance state loaded once per Mapper task (in setup)
    // ---------------------------------------------------------------
    private final Set<String> stopWords = new HashSet<>();
    private final Text word = new Text();

    // ---------------------------------------------------------------
    // setup() – called once before the first map() invocation.
    // Reads the stop-words file from the Distributed Cache into a
    // HashSet for O(1) look-ups inside map().
    // ---------------------------------------------------------------
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        // Retrieve the cached files that were added by the Driver
        URI[] cacheFiles = context.getCacheFiles();

        if (cacheFiles != null && cacheFiles.length > 0) {
            // The symlink name matches the filename we registered in the Driver
            Path stopWordsPath = new Path(cacheFiles[0].getPath());
            readStopWords(stopWordsPath.getName());   // uses local FS symlink
        }
    }

    /**
     * Reads every line of the stop-words file (one word per line, lower-case)
     * and populates the in-memory set.
     */
    private void readStopWords(String fileName) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                stopWords.add(line.trim().toLowerCase());
            }
        }
    }

    // ---------------------------------------------------------------
    // map() – tokenises the input line and emits (word, 1) for every
    // non-stop, alphabetic token.
    // ---------------------------------------------------------------
    @Override
    protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        // Split on any non-alphabetic character
        String[] tokens = value.toString().toLowerCase().split("[^a-zA-Z]+");

        for (String token : tokens) {
            // Skip empty strings, single characters, and stop-words
            if (token.isEmpty() || token.length() < 2) continue;
            if (stopWords.contains(token)) continue;

            word.set(token);
            context.write(word, ONE);
        }
    }
}
