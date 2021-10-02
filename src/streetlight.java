import java.lang.*;//provide classes that are fundamental to the design of the java programming language..

import weka.core.Instance;// importing weka..
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;// Via the ConverterUtils class it determines which converter to use for loading the data into memory
import weka.classifiers.bayes.NaiveBayes;//importing NaiveBayes..

public class streetlight { // this is a public class..

    public static void main(String[] args) throws Exception {
        //load training dataset
        DataSource source= new DataSource("C:\\Users\\91876\\Desktop\\dataset2.arff");// attaching data via code..
        Instances trainDataset= source.getDataSet();
        trainDataset.setClassIndex(trainDataset.numAttributes()-1);
        //built model
        NaiveBayes nb= new NaiveBayes();
        nb.buildClassifier(trainDataset);

        System.out.println(nb);

        DataSource source1= new DataSource("C:\\Users\\91876\\Desktop\\dataset2.arff");//load new dataset
        Instances testDataSet=source1.getDataSet();

        testDataSet.setClassIndex(testDataSet.numAttributes()-1);
        System.out.println("Actual Class , Predicted");

        for (int i=0;i<testDataSet.numInstances();i++)//loop through the new dataset and make predictions
        {
            double actualValue= testDataSet.instance(i).classValue();
            String actual = testDataSet.classAttribute().value((int)actualValue);
            System.out.println("actual ->"+actual);
            Instance newInst = testDataSet.instance(i);//get Instance object of current instance
            double predValue = nb.classifyInstance(newInst);//call classifyInstance, which returns a double value for the class
            String predString = testDataSet.classAttribute().value((int)predValue);
            System.out.println("PredString -> "+predString);
            System.out.println();
        }
    }

}
