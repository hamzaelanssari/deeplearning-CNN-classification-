package Diabetes;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class DiabetesApp {
    public static void main(String[] args) throws IOException, InterruptedException {
    int seed=123; double learningRate=0.001;
    int batchSize=1; int nEpochs=100;
    int numIn=8;int numOut=2; int nHidden=30;
    int classIndex=8;


        System.out.println("Model Creating");
    MultiLayerConfiguration multiLayerConfiguration=new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .list()
            .layer(0, new DenseLayer.Builder()
                    .nIn(numIn)
                    .nOut(nHidden)
                    .activation(Activation.SIGMOID)
                    .build()
            )
            .layer(1,new OutputLayer.Builder()
                    .nIn(nHidden)
                    .nOut(numOut)
                    .activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                    .build()
            )
            .build();
    MultiLayerNetwork model=new MultiLayerNetwork(multiLayerConfiguration);
    model.init();
        //System.out.println(multiLayerConfiguration.toJson());
        UIServer uiServer=UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));
        System.out.println("Model Training");

        String filePathTrain=new ClassPathResource("Diabetes/diabetes_train.csv").getFile().getPath();
        RecordReader rr=new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filePathTrain)));
        DataSetIterator dataSetTrain=new RecordReaderDataSetIterator(rr,batchSize,classIndex,numOut);
        while(dataSetTrain.hasNext()){
            System.out.println("--------------------------------------------");
            DataSet dataSet=dataSetTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }

       for(int i=0;i<nEpochs;i++) {
           model.fit(dataSetTrain);
       }


        System.out.println("Model Evaluation");
        String filePathTest=new ClassPathResource("Diabetes/diabetes_test.csv").getFile().getPath();
        RecordReader rrTest=new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filePathTest)));
        DataSetIterator dataSetTest=new RecordReaderDataSetIterator(rrTest,batchSize,classIndex,numOut);

        Evaluation evaluation=new Evaluation();
        while (dataSetTest.hasNext()){
            DataSet dataSet = dataSetTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labels=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());
        ModelSerializer.writeModel(model,"Diabetes_Model.zip",true);

    }}
