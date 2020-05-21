package Iris;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {
        String[] StringLabels={"Iris-setosa","Iris-versicolor","Iris-virginica"};
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("iris_Model.zip"));
        System.out.println("Prediction");
        INDArray inputData= Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2},
                {4.9,3.0,1.4,0.2},
                {6.7,3.1,1.4,1.4},
                {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        });
        INDArray output=model.output(inputData);
        System.out.println(output);

        int[] classes=output.argMax(1).toIntVector();
        for (int i=0;i<classes.length;i++){
            System.out.println("Class :"+StringLabels[classes[i]]);
        }
    }}
