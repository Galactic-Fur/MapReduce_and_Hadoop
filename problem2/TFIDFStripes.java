// TFIDFStripes.java
// Task B: MapReduce job to compute TF-IDF scores for the top-100 terms
// using the Stripes approach for Term Frequency.
//
// Formula : SCORE = TF × log(10000 / (DF + 1))
// Input   : docID <TAB> document-text   (same Wikipedia dump)
// Cache   : top-100 DF file (TERM <TAB> DF) produced by Task A
// Output  : ID <TAB> TERM <TAB> SCORE

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;

import java.io.*;
import java.net.URI;
import java.util.*;

public class TFIDFStripes {

    // Total number of documents assumed in the Wikipedia dump.
    // Adjust this constant to match the actual corpus size.
    private static final int N_DOCS = 10_000;

    // -----------------------------------------------------------------------
    // MAPPER  — Stripes approach
    //
    // setup()
    //   Reads the top-100 DF file from the Distributed Cache into a HashMap.
    //   This happens ONCE per task JVM, before any map() calls.
    //
    // map()
    //   Builds a "stripe" per document: a Map<term, rawTF> containing only
    //   terms that appear in the top-100 DF map.
    //   Emits: key = docID,  value = MapWritable stripe
    //
    // Why Stripes?
    //   Each (docID, stripe) record carries all relevant term counts for that
    //   document in a single shuffle value, reducing per-term shuffle overhead
    //   compared to the Pairs approach.
    // -----------------------------------------------------------------------
    public static class TFIDFMapper
            extends Mapper<Object, Text, Text, MapWritable> {

        // Loaded once in setup(); used in every map() call.
        private final Map<String, Integer> dfMap = new HashMap<>();

        private final PorterStemmer stemmer = new PorterStemmer();
        private final Text docIdKey = new Text();

        // ------------------------------------------------------------------
        // setup() — Distributed Cache access
        //
        // YARN copies the cache file to the task node's local working
        // directory before setup() is called.  context.getCacheFiles()
        // returns the URIs registered via job.addCacheFile().
        // The URI fragment (e.g. #df_results) becomes the symlink name.
        // ------------------------------------------------------------------
        @Override
        protected void setup(Context context)
                throws IOException, InterruptedException {

            URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles == null || cacheFiles.length == 0) {
                throw new IOException(
                    "Distributed Cache file not found. " +
                    "Ensure job.addCacheFile() was called with the DF output path.");
            }

            // Derive the local file name from the URI path.
            // If a fragment was provided (e.g. hdfs:///path/part-r-00000#df_results)
            // the fragment becomes the symlink name on the task node.
            String localName = new Path(cacheFiles[0].getPath()).getName();

            // Open the local copy using a plain FileReader — no HDFS client needed.
            try (BufferedReader br =
                         new BufferedReader(new FileReader(localName))) {

                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split("\t");
                    if (parts.length != 2) continue;        // skip blank/malformed

                    String term = parts[0].trim();
                    int    df   = Integer.parseInt(parts[1].trim());

                    // dfMap is pre-filtered to the top-100 terms BEFORE being
                    // uploaded to HDFS; we load all entries found in the file.
                    dfMap.put(term, df);
                }
            }

            // At this point dfMap contains at most 100 entries.
            // map() will only track terms found in this map.
        }

        // ------------------------------------------------------------------
        // map() — build per-document stripe
        // ------------------------------------------------------------------
        @Override
        protected void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line   = value.toString();
            int    tabIdx = line.indexOf('\t');
            if (tabIdx < 0) return;

            String docId   = line.substring(0, tabIdx).trim();
            String content = line.substring(tabIdx + 1).toLowerCase();

            // stripe: term → raw term count within this document
            Map<String, Integer> stripe = new HashMap<>();

            StringTokenizer tok = new StringTokenizer(
                    content, " \t\n\r\f.,;:!?\"'()-[]{}");

            while (tok.hasMoreTokens()) {
                String token = tok.nextToken();
                if (!token.matches("[a-z]+") || token.length() < 2) continue;

                // --- Porter Stemmer integration (same pattern as Task A) ---
                stemmer.stem(token);
                String stem = stemmer.toString();

                // Only count terms in the top-100 DF map
                if (!dfMap.containsKey(stem)) continue;

                stripe.merge(stem, 1, Integer::sum);
            }

            if (stripe.isEmpty()) return;   // document contains none of top-100

            // Serialise the stripe into a Hadoop MapWritable so it can be
            // shuffled to the reducer.
            MapWritable stripeWritable = new MapWritable();
            for (Map.Entry<String, Integer> e : stripe.entrySet()) {
                stripeWritable.put(
                        new Text(e.getKey()),
                        new org.apache.hadoop.io.IntWritable(e.getValue()));
            }

            docIdKey.set(docId);
            context.write(docIdKey, stripeWritable);
        }
    }

    // -----------------------------------------------------------------------
    // REDUCER
    //
    // setup()
    //   Re-loads the same DF cache file so the scoring formula can look up
    //   the DF for each term.
    //
    // reduce()
    //   Merges all stripes received for one docID (there may be multiple
    //   if the document was split across HDFS blocks).
    //   Computes SCORE = TF × log(N_DOCS / (DF + 1)) for each term.
    //   Emits: ID <TAB> TERM <TAB> SCORE
    // -----------------------------------------------------------------------
    public static class TFIDFReducer
            extends Reducer<Text, MapWritable, Text, Text> {

        private final Map<String, Integer> dfMap = new HashMap<>();

        @Override
        protected void setup(Context context)
                throws IOException, InterruptedException {

            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null || cacheFiles.length == 0) return;

            String localName = new Path(cacheFiles[0].getPath()).getName();

            try (BufferedReader br =
                         new BufferedReader(new FileReader(localName))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split("\t");
                    if (parts.length != 2) continue;
                    dfMap.put(parts[0].trim(),
                              Integer.parseInt(parts[1].trim()));
                }
            }
        }

        @Override
        protected void reduce(Text docId,
                              Iterable<MapWritable> stripes,
                              Context context)
                throws IOException, InterruptedException {

            // Merge all stripes for this document into one TF map
            Map<String, Integer> tfMap = new HashMap<>();

            for (MapWritable stripe : stripes) {
                for (Map.Entry<org.apache.hadoop.io.Writable,
                               org.apache.hadoop.io.Writable> e
                        : stripe.entrySet()) {

                    String term = e.getKey().toString();
                    int    cnt  = ((org.apache.hadoop.io.IntWritable)
                                        e.getValue()).get();
                    tfMap.merge(term, cnt, Integer::sum);
                }
            }

            // Emit one scored record per term
            for (Map.Entry<String, Integer> e : tfMap.entrySet()) {
                String term  = e.getKey();
                int    tf    = e.getValue();
                int    df    = dfMap.getOrDefault(term, 0);

                // SCORE = TF × log(N / (DF + 1))   (natural log)
                double score = tf * Math.log((double) N_DOCS / (df + 1));

                // Output schema: ID <TAB> TERM <TAB> SCORE
                context.write(docId,
                        new Text(term + "\t" + String.format("%.6f", score)));
            }
        }
    }

    // -----------------------------------------------------------------------
    // DRIVER
    // -----------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println(
                "Usage: TFIDFStripes <input-path> <df-cache-hdfs-uri> <output-path>");
            System.err.println(
                "  df-cache-hdfs-uri example: hdfs:///user/you/top100_df.txt#df_results");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "TF-IDF Stripes");

        job.setJarByClass(TFIDFStripes.class);
        job.setMapperClass(TFIDFMapper.class);
        job.setReducerClass(TFIDFReducer.class);

        // Mapper output: docID → MapWritable stripe
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);

        // Final output: Text key, Text value
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // ------------------------------------------------------------------
        // Distributed Cache registration
        //
        // addCacheFile() tells YARN to copy the HDFS file to every task node
        // before any map/reduce tasks start.  The URI fragment (#df_results)
        // becomes the local symlink name, which is what FileReader opens in
        // setup().
        //
        // args[1] should be something like:
        //   hdfs:///user/hadoop/df-out/part-r-00000#df_results
        // ------------------------------------------------------------------
        job.addCacheFile(new URI(args[1]));

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
