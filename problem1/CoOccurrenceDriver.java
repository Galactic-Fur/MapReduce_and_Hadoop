package task2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import task2.pairs.PairsMapper;
import task2.pairs.PairsReducer;
import task2.stripes.StripesMapper;
import task2.stripes.StripesReducer;

import java.net.URI;

/**
 * Task 2 Driver – runs either the Pairs or Stripes co-occurrence job.
 *
 * Usage:
 *   hadoop jar task2.jar task2.CoOccurrenceDriver \
 *       pairs|stripes <input> <output> <top50_vocab_hdfs> [window_size]
 *
 * The top-50 vocabulary file (output of Task 1 Job 2) is loaded into the
 * Distributed Cache so every Mapper can filter tokens locally.
 */
public class CoOccurrenceDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: CoOccurrenceDriver pairs|stripes <input> <output> " +
                "<vocab_hdfs_path> [window_size]"
            );
            return -1;
        }

        String approach   = args[0].toLowerCase();
        String inputPath  = args[1];
        String outputPath = args[2];
        String vocabHdfs  = args[3];
        int    windowSize = args.length > 4 ? Integer.parseInt(args[4]) : 2;

        // ------------------------------------------------------------------
        // Build Job
        // ------------------------------------------------------------------
        Configuration conf = getConf();
        String propKey = approach.equals("pairs")
                         ? "pairs.window.size" : "stripes.window.size";
        conf.setInt(propKey, windowSize);

        String jobName = "Task2-CoOccurrence-" + approach.toUpperCase();
        Job job = Job.getInstance(conf, jobName);
        job.setJarByClass(CoOccurrenceDriver.class);

        // Distributed Cache: top-50 vocabulary
        job.addCacheFile(new URI(vocabHdfs + "#top50.txt"));

        // ------------------------------------------------------------------
        // Mapper / Reducer wiring
        // ------------------------------------------------------------------
        if (approach.equals("pairs")) {
            job.setMapperClass(PairsMapper.class);
            job.setCombinerClass(PairsReducer.class);  // safe: sum is associative
            job.setReducerClass(PairsReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);

        } else if (approach.equals("stripes")) {
            job.setMapperClass(StripesMapper.class);
            // NOTE: A combiner for Stripes is also valid and recommended.
            //       The StripesReducer logic is identical for local merging.
            job.setCombinerClass(StripesReducer.class);
            job.setReducerClass(StripesReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

        } else {
            System.err.println("Unknown approach: " + approach +
                               ". Use 'pairs' or 'stripes'.");
            return -1;
        }

        // ------------------------------------------------------------------
        // I/O
        // ------------------------------------------------------------------
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CoOccurrenceDriver(), args);
        System.exit(exitCode);
    }
}
