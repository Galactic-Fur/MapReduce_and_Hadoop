// DocumentFrequency.java
// Task A: MapReduce job to compute Document Frequency (DF) for every term
// in the Wikipedia dump using Porter Stemmer and stop-word filtering.
//
// Input format  : docID <TAB> document-text   (one document per line)
// Output format : TERM <TAB> DF

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class DocumentFrequency {

    // -----------------------------------------------------------------------
    // Stop-word list (hardcoded for simplicity; could be loaded via cache)
    // -----------------------------------------------------------------------
    private static final Set<String> STOP_WORDS = new HashSet<>();
    static {
        String[] words = {
            "a","an","the","and","or","but","in","on","at","to","for",
            "of","with","by","from","is","was","are","were","be","been",
            "being","have","has","had","do","does","did","will","would",
            "could","should","may","might","shall","can","this","that",
            "these","those","it","its","i","we","you","he","she","they",
            "not","no","so","if","as","up","out","about","into","than",
            "then","more","also","which","who","what","when","where","how"
        };
        for (String w : words) STOP_WORDS.add(w);
    }

    // -----------------------------------------------------------------------
    // MAPPER
    //
    // For each document:
    //   1. Parse docID and content from the tab-delimited line.
    //   2. Tokenise, lowercase, remove non-alpha and stop-words.
    //   3. Stem each token with PorterStemmer.
    //   4. Emit (stem, 1) AT MOST ONCE per document (using a HashSet).
    //
    // Emitting only once per document means the reducer sum equals DF,
    // not total term count.
    // -----------------------------------------------------------------------
    public static class DFMapper extends Mapper<Object, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);

        // PorterStemmer is stateful but safe to reuse across tokens:
        // calling stemmer.stem(token) resets internal state before stemming.
        private final PorterStemmer stemmer = new PorterStemmer();
        private final Text termOut = new Text();

        @Override
        protected void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();

            // Expect format: "docID\tcontent"
            int tabIdx = line.indexOf('\t');
            if (tabIdx < 0) return;                        // skip malformed lines

            String content = line.substring(tabIdx + 1).toLowerCase();

            // Track unique stems seen in THIS document to ensure DF semantics
            Set<String> seenInDoc = new HashSet<>();

            StringTokenizer tok = new StringTokenizer(
                    content, " \t\n\r\f.,;:!?\"'()-[]{}");

            while (tok.hasMoreTokens()) {
                String token = tok.nextToken();

                // Keep only pure-alphabetic tokens of length >= 2
                if (!token.matches("[a-z]+") || token.length() < 2) continue;

                // Filter stop-words BEFORE stemming (cheaper check first)
                if (STOP_WORDS.contains(token)) continue;

                // --- Porter Stemmer integration ---
                // stem(token) feeds the token character-by-character internally,
                // then toString() returns the stemmed form.
                stemmer.stem(token);
                String stem = stemmer.toString();

                if (stem.isEmpty()) continue;

                seenInDoc.add(stem);
            }

            // Emit each unique stem exactly once for this document
            for (String stem : seenInDoc) {
                termOut.set(stem);
                context.write(termOut, ONE);
            }
        }
    }

    // -----------------------------------------------------------------------
    // REDUCER
    //
    // Sum the 1s emitted by mappers.  Because each mapper emits a term at most
    // once per document, the sum equals the Document Frequency (DF).
    //
    // Output: TERM <TAB> DF
    // -----------------------------------------------------------------------
    public static class DFReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable dfOut = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values,
                              Context context)
                throws IOException, InterruptedException {

            int df = 0;
            for (IntWritable val : values) df += val.get();

            dfOut.set(df);
            context.write(key, dfOut);   // e.g.  "algorithm\t4217"
        }
    }

    // -----------------------------------------------------------------------
    // DRIVER
    // -----------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: DocumentFrequency <input-path> <output-path>");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Document Frequency");

        job.setJarByClass(DocumentFrequency.class);
        job.setMapperClass(DFMapper.class);
        job.setReducerClass(DFReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Single reducer → single output file; convenient to pass as cache
        // to Task B without needing to merge multiple part files.
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
