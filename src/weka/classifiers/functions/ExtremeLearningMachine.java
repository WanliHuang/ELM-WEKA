/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ExtremeLearningMachine.java
 *    Copyright (C) Wanli Huang 2018-2019 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.functions;

import java.nio.charset.IllegalCharsetNameException;
import java.util.Random;

import no.uib.cipr.matrix.*;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.pmml.jaxbbindings.True;

/**
 * <!-- globalinfo-start -->
 *     Class for building and using Extreme Learning Machine (ELM) method <br/>
 *     Codes reference: https://github.com/StrongYeah/WEKA-ELM and https://github.com/ExtremeLearningMachines/ELM-JAVA <br/>
 * <br/>
 * For more details, please refer to the paper of Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Sew (2004) and Guang-Bin Huang, Hongming Zhou, Xiaojian Ding and Rui Zhang(2012)<br/>
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{G.B.Huang-ICNN,
 * author = {Huang, Guang-Bin and Zhu, Qin-Yu and Siew, Chee},
 * year = {2004},
 * month = {08},
 * pages = {985 - 990 vol.2},
 * booktitle={2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No.04CH37541)},
 * title = {Extreme learning machine: A new learning scheme of feedforward neural networks},
 * volume = {2},
 * isbn = {0-7803-8359-1},
 * journal = {IEEE International Conference on Neural Networks - Conference Proceedings},
 * doi = {10.1109/IJCNN.2004.1380068}
 * }
 * </pre>
 * <pre>
 *  &#64;ARTICLE{6035797,
 * author={G. {Huang} and H. {Zhou} and X. {Ding} and R. {Zhang}},
 * journal={IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)},
 * title={Extreme Learning Machine for Regression and Multiclass Classification},
 * year={2012},
 * volume={42},
 * number={2},
 * pages={513-529},
 * doi={10.1109/TSMCB.2011.2168604},
 * ISSN={1083-4419},
 * month={April},}
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -n
 *  number of hidden neuron unit.
 * </pre>
 *
 * <pre>
 * -t
 *  set the type of ELM.
 *  0 is for Regression,
 *  1 is for classification
 * </pre>
 *
 * <pre>
 * -a
 *  Set the activation function
 *  1 - Sigmoid function
 *  2 - Sin function
 *  3 - Hardlim function
 *  4 - Yibas function
 *  5 - Radbas function
 * </pre>
  <!-- options-end -->
 *
 * @author Wanli Huang
 * @email wanli.huang@gmail.com
 * @version $Revision$
 */
public class ExtremeLearningMachine extends AbstractClassifier {

    // Parameters that can be configured through Weka command or GUI
    protected int m_numHiddenNeurons = 20; // 20 hidden neurons by default

    protected int typeOfELM = 1;  // use ELM as a classifier by default

    protected int typeOfActivation = 1;  // use Sig Activation function by default

    protected int m_seed = -1; //random seed. if m_seed = -1 , don't use seed to generate random value
    //this is default case. By default, MTJ random matrix function is called

    protected int m_debug = 1;  // debugging mode switch

    // Option Metadata for Weka commands

    @OptionMetadata(
            displayName = "Hidden Neutrons",
            description = "Number of Neutrons",
            displayOrder = 1,
            commandLineParamName = "node",
            commandLineParamSynopsis = "-node"
    )
    public int getM_numHiddenNeurons(){
        return this.m_numHiddenNeurons;
    }
    public void setM_numHiddenNeurons(int number){
        this.m_numHiddenNeurons = number;
    }

    @OptionMetadata(
            displayName = "ELM type",
            description = "ELM type: 0 for regression, 1 for classification",
            displayOrder = 2,
            commandLineParamName = "type",
            commandLineParamSynopsis = "-type"
    )

    public int getTypeOfELM(){
        return this.typeOfELM;
    }

    public void setTypeOfELM(int type){
        this.typeOfELM = type;
    }

    @OptionMetadata(
            displayName = "Activation Function",
            description = "Choose Activate Function: " +
                    "  1 - Sig function, " +
                    "  2 - Sin Function, " +
                    "  3 - Hardlim function, " +
                    "  4 - Yibas function, " +
                    "  5 - Radbas function",
            displayOrder = 3,
            commandLineParamName = "activate",
            commandLineParamSynopsis = "-activate"
    )

    public int getTypeOfActivation(){
        return this.typeOfActivation;
    }
    public void setTypeOfActivation(int activationType){
        this.typeOfActivation = activationType;
    }

    @OptionMetadata(
            displayName = "Random Seed",
            description = "seed to be used for generating random number. -1 means using MTJ libraries's Matrices.random" +
                          "rather than using own generateRandomMatrix method",
            displayOrder = 4,
            commandLineParamName = "seed",
            commandLineParamSynopsis = "-seed"
    )
    public int getM_seed(){return this.m_seed;}
    public void setM_seed(int seed){this.m_seed = seed;}

    public String globalInfo() {

        return  "Extreme Learning Machine Weka version";
    }

    @OptionMetadata(
            displayName = "Debug Mode",
            description = "Switch of debuging. set 1 to switch on debug mode, set 0 to switch off debug mode",
            displayOrder = 5,
            commandLineParamName = "debug",
            commandLineParamSynopsis = "-debug"
    )
    public int getM_debug(){return this.m_debug;}
    public void setM_debug(int on) {this.m_debug = on;}



    // Other variables
    private DenseMatrix weightsOfInput;
    private DenseMatrix biases;
    private DenseMatrix weightsOfOutput;
    private DenseMatrix instancesMatrix;
    private DenseMatrix classesMatrix;
    private DenseMatrix testInstancesMatrix;
    private DenseMatrix testClassesMatrix;
    double[][] arrayMinMax; // to hold the minimum and maximum values for a numeric attribute

    private int m_numOfInputNeutrons;  // it is actually the number of attributes
    private int m_numOfOutputNeutrons = 1;  // it is actually the number of classes
    private int m_numOfInstances;
    //private int m_numOfTestInstances;

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }


    /**
     * Generates the classifier.
     *
     * @param rawdata set of instances serving as training data
     * @throws Exception if the classifier has not been generated successfully
     */
    @Override
    public void buildClassifier(Instances rawdata) throws Exception {
        // Determine whether the classifier can handle the data
        getCapabilities().testWithFail(rawdata);
        // Make a copy of data and delete instances with a missing class value
        Instances instances = new Instances(rawdata);
        instances.deleteWithMissingClass();
        if (m_debug == 1){
            System.out.println("the class index is: " + instances.classIndex());
        }



        if(typeOfELM == 0){
            m_numOfInputNeutrons = instances.numAttributes();
        }else if (typeOfELM == 1){
            m_numOfInputNeutrons = instances.numAttributes()-1;
        }else {
            System.out.println(" Wrong type of ELM. please check the parameter - type");
            System.exit(0);
        }
        m_numOfOutputNeutrons = instances.numClasses();

        if (instances.classAttribute().isNominal()) {
            if (typeOfELM != 1) {
                typeOfELM = 1;
                System.out.println("Setup wrong type of ELM. It should be a classifier because the class attribute is nominal. already reset ELM type to 1");
            }

        }

        if (typeOfELM == 0) m_numOfOutputNeutrons = 1;  // only one class for regression


        int numOfInstances = instances.numInstances();

        instancesMatrix = extractAttributes(instances);  //Matrix only containing features (attributes)
        if (m_debug ==1 ) {
            printMX("instanceMatrix", instancesMatrix);
        }

        classesMatrix = extractLabels(instances); //Matrix only containing classes (labels)
        if (m_debug ==1 ) {
            printMX("classesMatrix: ", classesMatrix);
        }
        if (m_seed != -1 ) {
            weightsOfInput = generateRandomMatrix(m_numHiddenNeurons,m_numOfInputNeutrons, m_seed);
        }else{
            weightsOfInput = (DenseMatrix) Matrices.random(m_numHiddenNeurons, m_numOfInputNeutrons);
        }


        biases =  (DenseMatrix)Matrices.random(m_numHiddenNeurons,1);

        DenseMatrix H = generateH(instancesMatrix,weightsOfInput,biases, numOfInstances);  // H , please refer to the paper of Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Sew (2004)
        if (m_debug ==1 ) {
            printMX("H", H);
        }

        weightsOfOutput = generateOutputWeights(H, classesMatrix, numOfInstances);  // this is the target that the model is trained for

        if (m_debug ==1 ) {
            printMX("weightsOfOutput", weightsOfOutput);
        }

    }

    /**
     * Classifies the given test instance. The instance has to belong to a dataset
     * when it's being classified. Note that a classifier MUST implement either
     * this or distributionForInstance().
     *
     * @param instance the instance to be classified
     * @return the predicted most likely class for the instance or
     *         Utils.missingValue() if no prediction is made
     * @exception Exception if an error occurred during the prediction
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {

        int classIndex = instance.classIndex();

        int numAttributes = 0;

        if (typeOfELM == 0) {
            numAttributes = instance.numAttributes(); // Regression
        }else  if (typeOfELM == 1){
            numAttributes = instance.numAttributes()-1;
        }else {
            System.out.println("Wrong type of ELM. please check the parameter - type");
            System.exit(0);
        }
        double[] testData = new double[numAttributes];

        if (classIndex == numAttributes || typeOfELM == 0) { //if the class is in the last column
            // normalize numeric value
            for (int i = 0; i < numAttributes; i++) {

                if (instance.attribute(i).isNumeric()) {

                    if (instance.value(i) <= arrayMinMax[1][i]) {
                        testData[i] = 0;
                    } else if (instance.value(i) >= arrayMinMax[0][i]) {
                        testData[i] = 1;
                    } else {
                        testData[i] = (instance.value(i) - arrayMinMax[1][i]) / (arrayMinMax[0][i] - arrayMinMax[1][i]);
                    }

                } else {
                    testData[i] = instance.value(i);
                }

            }
        }else if (classIndex == 0  && typeOfELM == 1){  // if the class attribute is in the first column
            // normalize numeric value
            for (int i = 1; i < numAttributes+1; i++) {

                if (instance.attribute(i).isNumeric()) {

                    if (instance.value(i) <= arrayMinMax[1][i-1]) {
                        testData[i-1] = 0;
                    } else if (instance.value(i) >= arrayMinMax[0][i-1]) {
                        testData[i-1] = 1;
                    } else {
                        testData[i-1] = (instance.value(i) - arrayMinMax[1][i-1]) / (arrayMinMax[0][i-1] - arrayMinMax[1][i-1]);
                    }

                } else {
                    testData[i-1] = instance.value(i);
                }

            }
        }
        DenseMatrix prediction = new DenseMatrix(numAttributes,1);
        for (int i = 0; i<numAttributes; i++){
            prediction.set(i, 0, testData[i]);
        }

        DenseMatrix H_test =  generateH(prediction,weightsOfInput,biases, 1);

        DenseMatrix H_test_T = new DenseMatrix(1, m_numHiddenNeurons);

        H_test.transpose(H_test_T);

        DenseMatrix output = new DenseMatrix(1, m_numOfOutputNeutrons);

        H_test_T.mult(weightsOfOutput, output);

        double result = 0;

        if (typeOfELM == 0) {
            result = output.get(0,0);
        }else if (typeOfELM == 1){
            int indexMax = 0;
            double labelValue = output.get(0,0);
            for (int i =0; i< m_numOfOutputNeutrons; i++){
                if (output.get(0,i) > labelValue){
                    labelValue = output.get(0,i);
                    indexMax = i;
                }
            }
            result = indexMax;
        }



        return result;


    }



//     /**
//     * Batch prediction method. This default implementation simply calls
//     * distributionForInstance() for each instance in the batch. If subclasses can
//     * produce batch predictions in a more efficient manner than this they should
//     * override this method and also return true from
//     * implementsMoreEfficientBatchPrediction()
//     *
//     * @param batch the instances to get predictions for
//     * @return an array of probability distributions, one for each instance in the
//     *         batch
//     * @throws Exception if a problem occurs.
//     */
//
//    public double[][] distributionForInstances(Instances batch) throws Exception {
//
//        int numOfClasses = batch.numClasses();
//
//
//        int numOfInstances = batch.numInstances();
//
//        double[][] ds=new double[numOfClasses][numOfInstances];
//
//        instancesMatrix = extractAttributes(batch);
//
//        classesMatrix = extractLabels(batch);
//
//
//        DenseMatrix testH = new DenseMatrix(m_numHiddenNeurons, numOfInstances);
//        testH = generateH(instancesMatrix,weightsOfInput,biases, numOfInstances);
//
//        DenseMatrix transposedTestH = new DenseMatrix(numOfInstances, m_numHiddenNeurons);
//        testH.transpose(transposedTestH);
//
//        DenseMatrix Predict = new DenseMatrix(numOfInstances, m_numOfOutputNeutrons);
//        transposedTestH.mult(weightsOfOutput, Predict);
//
//        DenseMatrix transposedPredict = new DenseMatrix(m_numOfOutputNeutrons, numOfInstances);
//        Predict.transpose(transposedPredict);
//        for (int i = 0; i < numOfClasses; i++){
//            for (int j = 0;j < numOfInstances; j++){
//                ds[i][j] = transposedPredict.get(i,j);
//            }
//        }
//
//        return ds;
//    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {
//        weka.gui.explorer.Explorer explorer=new weka.gui.explorer.Explorer();
//        String[] file={};
//        explorer.main(file);
        runClassifier(new ExtremeLearningMachine(), argv);
    }


    /**
     * Extract all attributes into a Matrix.
     *
     * @param instances the instances to be classified
     * @return a matrix containing all attributes (Dimension Row: Number of Attribute, Colunm: number of Instances)

     */
    private DenseMatrix extractAttributes(Instances instances){


        int classIndex = instances.classIndex();


        int numInstances = instances.numInstances();
        int numAttributes = 0;

        if (typeOfELM == 0) {
            numAttributes = instances.numAttributes(); // Regression
        }else  if (typeOfELM == 1){
            numAttributes = instances.numAttributes()-1;
        }else {
            System.out.println("Wrong type of ELM. please check the parameter type");
            System.exit(0);
        }

        DenseMatrix AttributesMatrix = new DenseMatrix(numAttributes, numInstances);



        // use formula (value - min)/(max-min)  to normalize value.

        arrayMinMax = new double[2][numAttributes];

        if (classIndex == instances.numAttributes()-1 || typeOfELM == 0) {
            // the class attribute is the last attribute
            for (int i = 0; i < numAttributes; i++) {

                if (instances.attribute(i).isNumeric()) {
                    arrayMinMax[0][i] = instances.attributeStats(i).numericStats.max;  // the maximum value of the attribute
                    arrayMinMax[1][i] = instances.attributeStats(i).numericStats.min;  // the minimum value of the attribute
                }

            }

            for (int i = 0; i < numAttributes; i++) {


                if (instances.attribute(i).isNumeric()) {
                    for (int j = 0; j < numInstances; j++) {
                        double tempValue = instances.instance(j).value(i) - arrayMinMax[1][i];
                        double normalizedValue = tempValue / (arrayMinMax[0][i] - arrayMinMax[1][i]);
                        AttributesMatrix.set(i, j, normalizedValue);
                    }
                } else {
                    for (int j = 0; j < numInstances; j++) {
                        AttributesMatrix.set(i, j, instances.instance(j).value(i));
                    }
                }


            }
        } else if (classIndex == 0 && typeOfELM == 1) {
           // the class attribute is the first attribute
            for (int i =1; i < numAttributes+1; i++) {

                if (instances.attribute(i).isNumeric()) {
                    arrayMinMax[0][i-1] = instances.attributeStats(i).numericStats.max;  // the maximum value of the attribute
                    arrayMinMax[1][i-1] = instances.attributeStats(i).numericStats.min;  // the minimum value of the attribute
                }

            }

            for (int i = 1; i < numAttributes+1; i++) {


                if (instances.attribute(i).isNumeric()) {
                    for (int j = 0; j < numInstances; j++) {
                        double tempValue = instances.instance(j).value(i) - arrayMinMax[1][i-1];
                        double normalizedValue = tempValue / (arrayMinMax[0][i-1] - arrayMinMax[1][i-1]);
                        AttributesMatrix.set(i-1, j, normalizedValue);
                    }
                } else {
                    for (int j = 0; j < numInstances; j++) {
                        AttributesMatrix.set(i-1, j, instances.instance(j).value(i));
                    }
                }


            }
        }else {
            System.out.println("Please put class attribute either on the first column or the last column");
            System.exit(0);
        }

        return AttributesMatrix;
    }

    /**
     * Extract all labels into a  Matrix.
     *
     * @param instances the instance to be classified
     * @return a matrix containing all labels (Dimension Row: Number of classes, Column: number of Instances)

     */

    private DenseMatrix extractLabels(Instances instances){

        int numInstances = instances.numInstances();
        int numClasses = instances.numClasses();
        int numAttributes = instances.numAttributes();
        double attMax = 0;
        double attMin = 0;
        if (typeOfELM == 0){
            numClasses = 1;
            attMax = instances.attributeStats(numAttributes-1).numericStats.max;  // the maximum value of the last attribute
            attMin = instances.attributeStats(numAttributes-1).numericStats.min;  // the minimum value of the last attribute

        }


        DenseMatrix LabelsMatrix = new DenseMatrix(numClasses, numInstances);


        for (int i = 0; i < numInstances; i++) {

            if (typeOfELM == 1 ) {

                for (int j = 0; j < numClasses; j++) {  //labels: 0, 1, 2 ......

                    LabelsMatrix.set(j, i, instances.instance(i).classValue() == j ? 1 : -1); // fill all non-label with -1

                }
            }else if (typeOfELM == 0){
                double normalizationValue = (instances.instance(i).value(numAttributes - 1) - attMin) / (attMax - attMin);  //normalize the firs attribute
                LabelsMatrix.set(0, i, normalizationValue);  // allocate the first attribute value to the label matrix
            }
        }




        return LabelsMatrix;
    }

    /**
     *
     * @param AttrMatrix
     * @param InputWeightsMatrix
     * @param Biases
     * @param numOfInstances
     * @return a matrix containing G(w*x + b)  dimension Row: number of hidden neutrons Column: number of instances

     */

    private DenseMatrix generateH(DenseMatrix AttrMatrix, DenseMatrix InputWeightsMatrix, DenseMatrix Biases, int numOfInstances) {

        DenseMatrix tempH = new DenseMatrix(m_numHiddenNeurons, numOfInstances);
        InputWeightsMatrix.mult(AttrMatrix, tempH);
        DenseMatrix BiasMX = new DenseMatrix(m_numHiddenNeurons, numOfInstances);

        for (int i = 0; i < numOfInstances; i++){
            for (int j = 0; j< m_numHiddenNeurons; j++){
                BiasMX.set(j, i, Biases.get(j,0));   // fill up each column (instance) with bias value;
            }
        }



        tempH.add(BiasMX);

        //DenseMatrix H = new DenseMatrix(m_numHiddenNeurons, numOfInstances);

        if (typeOfELM == 1){


            for (int i=0; i<m_numHiddenNeurons; i++){
                for (int j=0; j< numOfInstances; j++){
                    double v = Activation(tempH.get(i,j), typeOfActivation);
                    tempH.set(i,j, v);
                }
            }


        }else {
            // to do exception
        }

        return tempH;


    }

    /**
     * generate output weight matrix
     *
     * @param H
     * @param classesMatrix

     * @return a matrix containing Output weights  dimension Row: number of hidden neutrons Column: number of output neutrons

     */
    private DenseMatrix generateOutputWeights(DenseMatrix H, DenseMatrix classesMatrix, int numInstances) throws NotConvergedException {

        DenseMatrix HT = new DenseMatrix(numInstances,m_numHiddenNeurons);
        H.transpose(HT);
        Inverse inverseOfHT = new Inverse(HT);
        DenseMatrix MoorePenroseInvHT = inverseOfHT.getMPInverse();
        if (m_debug == 1){
            printMX("MoorePenroseInvHT", MoorePenroseInvHT);
        }
        DenseMatrix outputWeightsMX = new DenseMatrix(m_numHiddenNeurons, m_numOfOutputNeutrons);

        DenseMatrix TransposedClassesMX = new DenseMatrix(numInstances, m_numOfOutputNeutrons);

        classesMatrix.transpose(TransposedClassesMX);


        MoorePenroseInvHT.mult(TransposedClassesMX, outputWeightsMX);

        return  outputWeightsMX;

    }

    /**
     * Different activation functions
     *
     * @param value
     * @param Activation_type

     * @return return activation function's result

     */

    private double Activation(double value, int Activation_type){

        double result = 0.0;
        if (Activation_type  == 1){

            result = 1.0f / (1 + Math.exp(-value));

        }else if (Activation_type == 2){

            result = Math.sin(value);

        }else if (Activation_type == 3) {
            // to do
        }else if (Activation_type == 4) {
            // to do
        }else if (Activation_type == 5) {
            // to do
        }
        return result;
    }

    /**
     * Generate a matrix with random values
     *
     * @param rows
     * @param cols
     * @param m_seed

     * @return return activation function's result

     */
    private static DenseMatrix generateRandomMatrix (int rows, int cols, int m_seed){

        Random random_value = new Random(m_seed);
        DenseMatrix randomMX = new DenseMatrix(rows, cols);
        for (int i=0; i<rows; i++){
            for (int j=0; j<cols;j++){
                randomMX.set(i,j, random_value.nextDouble());
            }
        }
        return randomMX;
    }

    private void printMX (String nameMX, DenseMatrix MX){

        System.out.println(nameMX + "numRows: "+ MX.numRows());
        System.out.println(nameMX + "numColumns: "+ MX.numColumns());
        for (int i=0; i<MX.numRows(); i++){
            for (int j=0; j<MX.numColumns();j++){
                System.out.print(MX.get(i,j)+", ");
            }
            System.out.println("//");
        }

        System.out.println("Press Enter to continue");
        try{System.in.read();}
        catch(Exception e){}

    }





}
