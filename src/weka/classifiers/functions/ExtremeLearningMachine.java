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
 *    Copyright (C) Wanli Huang 2018-2019
 *
 */

package weka.classifiers.functions;

import java.util.Random;

import no.uib.cipr.matrix.*;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

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
 * -node
 *  number of hidden neuron unit.
 * </pre>
 *
 * <pre>
 * -type
 *  set the type of ELM.
 *  0 is for Regression,
 *  1 is for classification
 * </pre>
 *
 * <pre>
 * -activate
 *  Set the activation function
 *  1 - Sigmoid function
 *  2 - Sin function
 *  3 - Hardlim function
 *  4 - Yibas function
 *  5 - Radbas function
 * </pre>
 * * <pre>
 *  * -seed
 *  *  seed to generate random value
 *  *  -1 is to use default method
 *  * can be any integer
 *  * </pre>
 * <pre>
 * -debug
 *  debug mode switch
 *  0 is for debug mode off
 *  1 is for debug mode on
 * </pre>
  <!-- options-end -->
 *
 * @author Wanli Huang (wanli.huang@gmail.com)
 * @version $Revision$
 */
public class ExtremeLearningMachine extends AbstractClassifier {

    // Parameters that can be configured through Weka command or GUI
    private int m_numHiddenNeurons = 20; // 20 hidden neurons by default

    private int m_typeOfELM = 1;  // use ELM as a classifier by default

    private int m_typeOfActivation = 1;  // use Sig Activation function by default

    private int m_seed = -1; //random seed. if m_seed = -1 , don't use seed to generate random value
    //this is default case. By default, MTJ random matrix function is called

    private int m_debug = 1;  // debugging mode switch



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

    public int getM_typeOfELM(){
        return this.m_typeOfELM;
    }

    public void setM_typeOfELM(int type){
        this.m_typeOfELM = type;
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

    public int getM_typeOfActivation(){
        return this.m_typeOfActivation;
    }
    public void setM_typeOfActivation(int activationType){
        this.m_typeOfActivation = activationType;
    }

    @OptionMetadata(
            displayName = "Random Seed",
            description = "seed to be used for generating random number.",
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

    private Instances instances;

    private String[] labels;
    private int classIndex;

    private double[][] m_normalization ; //to hold max and min value for each numeric attribute

    private int m_numOfInputNeutrons;  // it is actually the number of attributes
    private int m_numOfOutputNeutrons = 1;  // it is actually the number of classes



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
        Instances instances = new Instances(rawdata);
        instances.deleteWithMissingClass();

        if (m_debug == 1) {
            printInstances("instances before normalizing",instances);
        }

        m_normalization = new double[2][instances.numAttributes()];
        // for each attribute | numeric attribute
        for (int i = 0; i < instances.numAttributes(); i++) {
            if (instances.attribute(i).isNumeric()) {
                m_normalization[0][i] = instances.attributeStats(i).numericStats.max;
                m_normalization[1][i] = instances.attributeStats(i).numericStats.min;
            }
        }

        int rows = instances.numInstances();
        int columns = instances.numAttributes();

        //normalization x-min/(max-min)

        for (int j = 0; j < columns; j++) {
            if (instances.attribute(j).isNumeric()) {
                double min = m_normalization[1][j];
                double max = m_normalization[0][j];
                double normValue = 0;
                for (int i = 0; i < rows; i++) {
                    if ( max  == min ){
                        normValue = max;
                    }else {
                       normValue = (instances.instance(i).value(j) - min)/(max - min);
                    }
                    instances.instance(i).setValue(j,normValue);
                }
            }
        }



        if (m_debug == 1) {
            printInstances("instances after normalizing",instances);
        }

        classIndex = instances.classIndex();
        if (m_debug == 1){
            System.out.println("the class index is: " + classIndex);
        }



        m_numOfInputNeutrons = instances.numAttributes()-1;

        m_numOfOutputNeutrons = instances.numClasses();


        if (instances.classAttribute().isNominal()) {
            if (m_typeOfELM != 1) {
                m_typeOfELM = 1;
                System.out.println("Setup wrong type of ELM. It should be a classifier because the class attribute is nominal. already reset ELM type to 1");
            }

        }

        if (m_typeOfELM == 0) m_numOfOutputNeutrons = 1;  // only one class for regression


        int numOfInstances = instances.numInstances();

        instancesMatrix = extractAttributes(instances);  //Matrix only containing features (attributes)
        if (m_debug ==1 ) {
            printMX("instanceMatrix", instancesMatrix);
        }

        classesMatrix = extractLabels(instances); //Matrix only containing classes (labels)
        if (m_debug ==1 ) {
            printMX("classesMatrix: ", classesMatrix);
        }


        weightsOfInput = generateRandomMatrix(m_numHiddenNeurons,m_numOfInputNeutrons, m_seed);
        if (m_debug ==1 ) {
            printMX("Random Input Weights: ", weightsOfInput);
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



        int numAttributes = instance.numAttributes();
        int clsIndex = instance.classIndex();
        boolean hasClassAttribute = true;
        int numTestAtt = numAttributes -1;
        if (numAttributes == m_numOfInputNeutrons) {
            hasClassAttribute = false; // it means the test data doesn't has class attribute
            numTestAtt = numTestAtt+1;
        }

        for (int i = 0; i< numAttributes; i++){
            if (instance.attribute(i).isNumeric()){

                double max = m_normalization[0][i];
                double min = m_normalization[1][i];
                double normValue = 0 ;
                if (instance.value(i)<min) {
                    normValue = 0;
                    m_normalization[1][i] = instance.value(i); // reset the smallest value
                }else if(instance.value(i)> max){
                    normValue = 1;
                    m_normalization[0][i] = instance.value(i); // reset the biggest value
                }else {
                    if (max == min ){
                        if (max == 0){
                            normValue = 0;
                        }else {
                            normValue = max/Math.abs(max);
                        }
                    }else {
                        normValue = (instance.value(i) - min) / (max - min);
                    }
                }
                instance.setValue(i, normValue);
            }
        }

        double[] testData = new double[numTestAtt];





        int index = 0 ;

        if (!hasClassAttribute){

            for (int i =0; i<numAttributes; i++) {
                    testData[i] = instance.value(i);
            }
        }else {
            for (int i = 0; i < numAttributes; i++) {

                if (i != clsIndex) {

                    testData[index] = instance.value(i);

                    index++;
                }
            }
        }



        DenseMatrix prediction = new DenseMatrix(numTestAtt,1);
        for (int i = 0; i<numTestAtt; i++){
            prediction.set(i, 0, testData[i]);
        }

        DenseMatrix H_test =  generateH(prediction,weightsOfInput,biases, 1);

        DenseMatrix H_test_T = new DenseMatrix(1, m_numHiddenNeurons);

        H_test.transpose(H_test_T);

        DenseMatrix output = new DenseMatrix(1, m_numOfOutputNeutrons);

        H_test_T.mult(weightsOfOutput, output);

        double result = 0;

        if (m_typeOfELM == 0) {
            double value = output.get(0,0);
            result = value*(m_normalization[0][classIndex]-m_normalization[1][classIndex])+m_normalization[1][classIndex];
            //result = value;
            if (m_debug == 1){
                System.out.print(result + " ");
            }
        }else if (m_typeOfELM == 1){
            int indexMax = 0;
            double labelValue = output.get(0,0);

            if (m_debug == 1){
                System.out.println("Each instance output neuron result (after activation)");
            }
            for (int i =0; i< m_numOfOutputNeutrons; i++){
                if (m_debug == 1){
                    System.out.print(output.get(0,i) + " ");
                }
                if (output.get(0,i) > labelValue){
                    labelValue = output.get(0,i);
                    indexMax = i;
                }
            }
            if (m_debug == 1){

                System.out.println("//");
                System.out.println(indexMax);
            }
            result = indexMax;
        }



        return result;


    }




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





        int numInstances = instances.numInstances();
        int numAttributes = instances.numAttributes();

        DenseMatrix AttributesMatrix = new DenseMatrix(numAttributes-1, numInstances); // except for classAttribute


        int index = 0; // index of attributeMatrix's row
        for (int i =0; i<numAttributes; i++){

            if (i != classIndex) {

                for (int j = 0; j < numInstances; j++) {
                    AttributesMatrix.set(index, j, instances.instance(j).value(i));

                }

                index++;
            }

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


        Attribute classAtt = instances.classAttribute();
        labels = new String[numClasses];


        if (m_typeOfELM == 1) {
            for (int i = 0; i < numClasses; i++) {

                labels[i] = classAtt.value(i);
                if (m_debug == 1){
                    System.out.print(labels[i]+", ");
                }
            }
        }


        if (m_typeOfELM == 0) numClasses = 1;



        DenseMatrix LabelsMatrix = new DenseMatrix(numClasses, numInstances);


        for (int i = 0; i < numInstances; i++) {

            if (m_typeOfELM == 1){
                for (int j = 0; j < numClasses; j++) {  //labels: 0, 1, 2 ......

                    String label = instances.instance(i).stringValue(classIndex);


                    LabelsMatrix.set(j, i, label.equals(labels[j]) ? 1 : -1); // fill all non-label with -1
                }
            }else if (m_typeOfELM == 0){
                LabelsMatrix.set(0,i,instances.instance(i).value(classIndex));
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

        if (m_typeOfELM == 1){


            for (int i=0; i<m_numHiddenNeurons; i++){
                for (int j=0; j< numOfInstances; j++){
                    double v = Activation(tempH.get(i,j), m_typeOfActivation);
                    tempH.set(i,j, v);
                }
            }


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
        Inverse inverseOfHT = new Inverse(HT, m_seed);
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
        if (Activation_type  == 1){    //Sig

            result = 1.0f / (1 + Math.exp(-value));

        }else if (Activation_type == 2){  //Sin

            result = Math.sin(value);

        }else if (Activation_type == 3) { //Hardlim
            // to do
        }else if (Activation_type == 4) { //Yibas
            // to do
        }else if (Activation_type == 5) { //Radbas
            double a = 2, b = 2, c = Math.sqrt(2);
            result =  a * Math.exp(-(value - b) * (value - b) / c * c);
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

    private void printInstances (String nameMX, Instances MX){

        System.out.println(nameMX + "numInstances "+ MX.numInstances());
        System.out.println(nameMX + "numAttributes: "+ MX.numAttributes());
        for (int i=0; i<MX.numInstances(); i++){
            for (int j=0; j<MX.numAttributes();j++){
                System.out.print(MX.instance(i).value(j)+", ");
            }
            System.out.println("//");
        }

        System.out.println("Press Enter to continue");
        try{System.in.read();}
        catch(Exception e){}

    }


}
