package Diabetes;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class DiabetesPrediction {
    public static void main(String[] args) throws IOException {
        String[] StringLabels={"Diabetic","Not Diabetic"};
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("Diabetes_Model.zip"));
        System.out.println("Prediction");
        INDArray inputData= Nd4j.create(new double[][]{
                {6,89,90,61,0,30.9,0.115,23},
                {10,114,58,0,0,37.8,0.247,90},
                {3,77,58,34,67,37.6,0.401,54}
        });
        INDArray output=model.output(inputData);
        System.out.println(output);

        int[] classes=output.argMax(1).toIntVector();
        for (int i=0;i<classes.length;i++){
            System.out.println("Class :"+StringLabels[classes[i]]);
        }
    }}
