package task1;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.net.URI;

/**
 * Task 1 Driver – chains two MapReduce jobs:
 *
 *   Job 1: WordCount  →  (word, totalCount) for every non-stop word
 *   Job 2: TopN       →  top-50 words by frequency
 *
 * Usage:
 *   hadoop jar task1.jar task1.WordCountDriver \
 *       <input_path> <intermediate_output> <final_output> <stopwords_hdfs_path>
 *
 * The stop-words file is registered with the Distributed Cache so every
 * Mapper in Job 1 can access it locally without HDFS reads per record.
 */
public class WordCountDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: WordCountDriver <input> <intermediate> <output> <stopwords>"
            );
            return -1;
        }

        String inputPath        = args[0];
        String intermediatePath = args[1];
        String outputPath       = args[2];
        String stopWordsHdfs    = args[3];   // e.g. hdfs:///data/stopwords.txt

        // ==================================================================
        // Job 1 – Word Count with stop-word filtering
        // ==================================================================
        Configuration conf1 = getConf();
        Job job1 = Job.getInstance(conf1, "Task1-WordCount");
        job1.setJarByClass(WordCountDriver.class);

        // --- Distributed Cache: ship the stop-words file to each node -----
        // The second argument creates a symlink so the Mapper can open it by
        // its local filename rather than a full HDFS URI.
        job1.addCacheFile(new URI(stopWordsHdfs + "#stopwords.txt"));

        // --- Classes -------------------------------------------------------
        job1.setMapperClass(WordCountMapper.class);
        job1.setCombinerClass(WordCountReducer.class);   // local aggregation
        job1.setReducerClass(WordCountReducer.class);

        // --- Output types --------------------------------------------------
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        // --- I/O formats ---------------------------------------------------
        job1.setInputFormatClass(TextInputFormat.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(intermediatePath));

        // Run Job 1 and wait for completion before starting Job 2
        if (!job1.waitForCompletion(true)) {
            System.err.println("Job 1 (WordCount) failed.");
            return 1;
        }

        // ==================================================================
        // Job 2 – Top-50 selection
        // ==================================================================
        Configuration conf2 = getConf();
        Job job2 = Job.getInstance(conf2, "Task1-Top50");
        job2.setJarByClass(WordCountDriver.class);

        // --- Classes -------------------------------------------------------
        job2.setMapperClass(TopNMapper.class);
        job2.setReducerClass(TopNReducer.class);
        job2.setNumReduceTasks(1);              // single reducer for global top-N

        // --- Output types --------------------------------------------------
        job2.setMapOutputKeyClass(NullWritable.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        // --- I/O formats ---------------------------------------------------
        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job2, new Path(intermediatePath));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath));

        return job2.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new WordCountDriver(), args);
        System.exit(exitCode);
    }
}
