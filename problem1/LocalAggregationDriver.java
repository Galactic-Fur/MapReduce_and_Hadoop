package task3;

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

import task2.pairs.PairsReducer;
import task2.stripes.StripesReducer;

import java.net.URI;

/**
 * Task 3 Driver – selects In-Mapper Combining variant at runtime.
 *
 * Usage:
 *   hadoop jar task3.jar task3.LocalAggregationDriver \
 *       pairs-map|pairs-class|stripes-map|stripes-class \
 *       <input> <o> <top50_vocab_hdfs> [window_size]
 *
 * Four variants are supported:
 *   pairs-map     → PairsMapFunctionLevelMapper    (map-function level)
 *   pairs-class   → PairsClassLevelMapper          (map-class level / IMC)
 *   stripes-map   → StripesMapFunctionLevelMapper  (map-function level)
 *   stripes-class → StripesClassLevelMapper        (map-class level / IMC)
 *
 * All variants share the same Reducer from Task 2 (Pairs or Stripes).
 */
public class LocalAggregationDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: LocalAggregationDriver " +
                "pairs-map|pairs-class|stripes-map|stripes-class " +
                "<input> <o> <vocab_hdfs> [window_size]"
            );
            return -1;
        }

        String variant    = args[0].toLowerCase();
        String inputPath  = args[1];
        String outputPath = args[2];
        String vocabHdfs  = args[3];
        int    windowSize = args.length > 4 ? Integer.parseInt(args[4]) : 2;

        boolean isPairs   = variant.startsWith("pairs");
        boolean isClass   = variant.endsWith("class");

        Configuration conf = getConf();
        String propKey = isPairs ? "pairs.window.size" : "stripes.window.size";
        conf.setInt(propKey, windowSize);

        Job job = Job.getInstance(conf, "Task3-IMC-" + variant.toUpperCase());
        job.setJarByClass(LocalAggregationDriver.class);
        job.addCacheFile(new URI(vocabHdfs + "#top50.txt"));

        // ---- Mapper selection ----------------------------------------
        if      (variant.equals("pairs-map"))     job.setMapperClass(PairsMapFunctionLevelMapper.class);
        else if (variant.equals("pairs-class"))   job.setMapperClass(PairsClassLevelMapper.class);
        else if (variant.equals("stripes-map"))   job.setMapperClass(StripesMapFunctionLevelMapper.class);
        else if (variant.equals("stripes-class")) job.setMapperClass(StripesClassLevelMapper.class);
        else {
            System.err.println("Unknown variant: " + variant);
            return -1;
        }

        // ---- Reducer selection ----------------------------------------
        // NOTE: For class-level (IMC) mappers we still use a Reducer because
        // multiple mappers run in parallel on different splits; their per-split
        // aggregates need to be combined globally.
        // We do NOT attach a Combiner for class-level mappers — the IMC pattern
        // already performs local combining more efficiently than the framework
        // Combiner (which would add unnecessary serialisation overhead).
        if (isPairs) {
            job.setReducerClass(PairsReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
        } else {
            job.setReducerClass(StripesReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
        }

        // For map-function level variants, a Combiner is still useful because
        // they emit once per unique pair/word per LINE, not once per SPLIT.
        if (!isClass) {
            if (isPairs)   job.setCombinerClass(PairsReducer.class);
            else           job.setCombinerClass(StripesReducer.class);
        }

        // ---- I/O -------------------------------------------------------
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new LocalAggregationDriver(), args));
    }
}
