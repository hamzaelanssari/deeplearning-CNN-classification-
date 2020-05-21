package ModelAlphabets;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class AlphabetsCNN {

    public static void main(String[] args) throws IOException, InterruptedException {

        long seed = 1234;
        double learningRate = 0.14;
        long height = 28;
        long width = 28;
        long depth = 1;
        int outputSize = 28;
        int batchSize = 50;
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Sgd(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height, width, depth))
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(depth)
                        .nOut(56)
                        .activation(Activation.RELU)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nOut(80)
                        .activation(Activation.RELU)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(600)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();

        System.out.println(config.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        System.out.println("Model training");

        File fileTrain = new File("src/main/resources/Arabic_Alphabets/train");
        FileSplit fileSplitTrain = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTrain = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrain);

        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputSize);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIteratorTrain.setPreProcessor(scaler);

        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));


        int numEpoch = 1;
        for (int i = 0; i < numEpoch; i++) {
            model.fit(dataSetIteratorTrain);
        }
        while (dataSetIteratorTrain.hasNext()){
            DataSet dataSet = dataSetIteratorTrain.next();
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            System.out.println(features.shapeInfoToString());
            System.out.println(labels.shapeInfoToString());
            System.out.println("---------------------------------------");
        }


        System.out.println("Model evaluation");

        File fileTest = new File("src/main/resources/Arabic_Alphabets/test");
        FileSplit fileSplitTest = new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTest = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, outputSize);
        DataNormalization scalerTest = new ImagePreProcessingScaler();
        dataSetIteratorTest.setPreProcessor(scalerTest);

        Evaluation evaluation = new Evaluation();
        while (dataSetIteratorTest.hasNext()) {
            DataSet dataSet=dataSetIteratorTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray targetlabels = dataSet.getLabels();
            INDArray predicted = model.output(features);

            evaluation.eval(predicted,targetlabels);
        }

        System.out.println(evaluation.stats());

        ModelSerializer.writeModel(model,"ModelNumbers.AlphabetsCNN.zip",true);

    }
}