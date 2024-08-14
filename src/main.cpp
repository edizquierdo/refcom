#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const int LN = 4;
const double StepSize = 0.1;
const double RunDuration = 300.0;
const double TransDuration = 150.0;
const double MinLength = 50.0;    // Minimum length of the 1-D field for positioning landmarks
const double mindist = 5.0;        // XXX Most recent change

// EA params
const int POPSIZE = 96;
const int GENS = 10000;
const double MUTVAR = 0.05;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 4;
const double WR = 10.0;     // Change from 8.0 to 10.0
const double SR = 10.0;     // Change from 8.0 to 10.0
const double BR = 10.0;     // Change from 8.0 to 10.0
const double TMIN = 1.0;
const double TMAX = 16.0;   // Change from 10.0 to 16.0

// Genotype size
int    VectSize = N*N + 4*N;

// ================================================
// A. FUNCTIONS FOR EVOLVING A SUCCESFUL CIRCUIT
// ================================================

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
        k++;
    }
    // Bias
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -BR, BR);
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
            //if (fabs(phen(k)) < 2.0){   // XXX
            //    phen(k) = 0.0;
            //}
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        // if (fabs(phen(k)) < 2.0){   // XXX
        //     phen(k) = 0.0;
        // }        
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        // if (fabs(phen(k)) < 2.0){   // XXX
        //     phen(k) = 0.0;
        // }        
        k++;
    }
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionA(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    //double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc;
    double fit;
    double ref = 15;
    double sep = 15;
    
    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // LEARNING PHASE
        // Reset agent's position and neural state
        Agent.ResetPosition(0);
        Agent.ResetNeuralState();

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(ref,sep,LN,env);
            // Move
            Agent.Step(StepSize);
        }

        // TESTING PHASE
        // ReSet agent's positions
        Agent.ResetPosition(0);

        // Set average distance to 0
        totaldist = 0.0;
        totaltime = 0.0;

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(ref,sep,LN,-1); // -1 food position so that it cannot sense it

            // Move
            Agent.Step(StepSize);

            // Measure distance between them (after transients)
            if (time > TransDuration)
            {
                loc = ref + env*sep;
                dist = fabs(Agent.pos - loc);
                if (dist < mindist){
                    dist = 0.0;
                }
                totaldist += dist;
                totaltime += 1;
            }
        }
        fit = 1 - ((totaldist / totaltime)/MinLength);
        if (fit < 0){
            fit = 0;
        }
        totalfit += fit;
        totaltrials += 1;
    }
    return totalfit/totaltrials;
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionB(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc;
    double fit;

    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // LEARNING PHASE
        // Reset agent's position and neural state
        Agent.ResetPosition(0);
        Agent.ResetNeuralState();

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(15,15,LN,env);
            // Move
            Agent.Step(StepSize);
        }

        // Save neural state
        for (int i = 1; i <= N; i++)
        {
            savedstate[i] = Agent.NervousSystem.NeuronState(i);
        }
        savedLS = Agent.landmarkSensor;
        savedFS = Agent.foodSensor;

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref = 14.0; ref <= 16.0; ref += 1.0)
        {
            for (double sep = 14.0; sep <= 16.0; sep += 1.0)
            {
                // TESTING PHASE
                // ReSet agent's positions
                Agent.ResetPosition(0);

                // Reset state
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, savedstate[i]);
                }
                Agent.landmarkSensor = savedLS;
                Agent.foodSensor = savedFS;
                Agent.Step(StepSize);

                // Set average distance to 0
                totaldist = 0.0;
                totaltime = 0.0;

                // Run sim
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    // Sense
                    Agent.Sense(ref,sep,LN,-1); // -1 food position so that it cannot sense it

                    // Move
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        loc = ref + env*sep;
                        dist = fabs(Agent.pos - loc);
                        if (dist < mindist){
                            dist = 0.0;
                        }
                        totaldist += dist;
                        totaltime += 1;
                    }
                }
                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0){
                    fit = 0;
                }
                totalfit += fit;
                totaltrials += 1;
            }
        }
    }
    return totalfit/totaltrials;
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionC(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc;
    double fit;

    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // LEARNING PHASE
        // Reset agent's position and neural state
        Agent.ResetPosition(0);
        Agent.ResetNeuralState();

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(15,15,LN,env);

            // Move
            Agent.Step(StepSize);
        }

        // Save neural state
        for (int i = 1; i <= N; i++)
        {
            savedstate[i] = Agent.NervousSystem.NeuronState(i);
        }
        savedLS = Agent.landmarkSensor;
        savedFS = Agent.foodSensor;

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (double ref = 13.0; ref <= 17.0; ref += 1.0)
        {
            for (double sep = 13.0; sep <= 17.0; sep += 1.0)
            {
                // TESTING PHASE
                // ReSet agent's positions
                Agent.ResetPosition(0);

                // Reset state
                for (int i = 1; i <= N; i++)
                {
                    Agent.NervousSystem.SetNeuronState(i, savedstate[i]);
                }
                Agent.landmarkSensor = savedLS;
                Agent.foodSensor = savedFS;
                Agent.Step(StepSize);

                // Set average distance to 0
                totaldist = 0.0;
                totaltime = 0.0;

                // Run sim
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    // Sense
                    Agent.Sense(ref,sep,LN,-1); // -1 food position so that it cannot sense it

                    // Move
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        loc = ref + env*sep;
                        dist = fabs(Agent.pos - loc);
                        if (dist < mindist){
                            dist = 0.0;
                        }
                        totaldist += dist;
                        totaltime += 1;
                    }
                }
                //totalfit += 1 - ((totaldist / totaltime)/MinLength);
                fit = 1 - ((totaldist / totaltime)/MinLength);
                if (fit < 0){
                    fit = 0;
                }
                totalfit += fit;
                totaltrials += 1;
            }
        }
    }
    return totalfit/totaltrials;
    //return totalfit;
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionD(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc;
    double fit;

    // Use this to save the neural state during learning
    for (int delay = 0; delay <= 10; delay += 5)
    {
        for (int env = 1; env <= LN; env += 1)
        {
            // LEARNING PHASE
            // Reset agent's position and neural state
            Agent.ResetPosition(0);
            Agent.ResetNeuralState();

            // Run sim
            for (double time = 0; time < RunDuration + delay; time += StepSize)
            {
                // Sense
                Agent.Sense(15,15,LN,env);
                // Move
                Agent.Step(StepSize);
            }

            // Save neural state
            for (int i = 1; i <= N; i++)
            {
                savedstate[i] = Agent.NervousSystem.NeuronState(i);
            }
            savedLS = Agent.landmarkSensor;
            savedFS = Agent.foodSensor;

            // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
            for (double ref = 13.0; ref <= 17.0; ref += 1.0)
            {
                for (double sep = 13.0; sep <= 17.0; sep += 1.0)
                {
                    // TESTING PHASE
                    // ReSet agent's positions
                    Agent.ResetPosition(0);

                    // Reset state
                    for (int i = 1; i <= N; i++)
                    {
                        Agent.NervousSystem.SetNeuronState(i, savedstate[i]);
                    }
                    Agent.landmarkSensor = savedLS;
                    Agent.foodSensor = savedFS;
                    Agent.Step(StepSize);

                    // Set average distance to 0
                    totaldist = 0.0;
                    totaltime = 0.0;

                    // Run sim
                    for (double time = 0; time < RunDuration; time += StepSize)
                    {
                        // Sense
                        Agent.Sense(ref,sep,LN,-1); // -1 food position so that it cannot sense it
                        // Move
                        Agent.Step(StepSize);

                        // Measure distance between them (after transients)
                        if (time > TransDuration)
                        {
                            loc = ref + env*sep;
                            dist = fabs(Agent.pos - loc);
                            if (dist < mindist){
                                dist = 0.0;
                            }
                            totaldist += dist;
                            totaltime += 1;
                        }
                    }
                    //totalfit += 1 - ((totaldist / totaltime)/MinLength);
                    fit = 1 - ((totaldist / totaltime)/MinLength);
                    if (fit < 0){
                        fit = 0;
                    }
                    totalfit += fit;
                    totaltrials += 1;
                }
            }
        }
    }
    return totalfit/totaltrials;
}

// ------------------------------------
// Behavior
// ------------------------------------
void BehaviorSimple(TVector<double> &genotype)
{
    ofstream perffile("perfsimple.dat");

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc, ref;

    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        ofstream pos1file("posLearn_"+std::to_string(env)+".dat");
        // LEARNING PHASE
        // Reset agent's position and neural state
        Agent.ResetPosition(0);
        Agent.ResetNeuralState();
        // Record position
        pos1file << Agent.pos << " ";
        // Record neural state
        ofstream neural1file("neuralLearn"+std::to_string(env)+".dat");
        for (int i = 1; i <= N; i++)
            neural1file << Agent.NervousSystem.NeuronOutput(i) << " ";
        neural1file << endl;

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(15,15,LN,env);
            // Move
            Agent.Step(StepSize);
            // Record position
            pos1file << Agent.pos << " ";
            // Record neural state
            for (int i = 1; i <= N; i++)
                neural1file << Agent.NervousSystem.NeuronOutput(i) << " ";
            neural1file << endl;
        }
        pos1file << endl;
        neural1file.close();

        // Save neural state
        for (int i = 1; i <= N; i++)
        {
            savedstate[i] = Agent.NervousSystem.NeuronState(i);
        }
        savedLS = Agent.landmarkSensor;
        savedFS = Agent.foodSensor;

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        ofstream pos2file("posTest_"+std::to_string(env)+".dat");
        int k = 0;
        // TESTING PHASE
        // ReSet agent's positions
        Agent.ResetPosition(0);

        // Record position
        pos2file << Agent.pos << " ";

        // Reset state
        for (int i = 1; i <= N; i++)
        {
            Agent.NervousSystem.SetNeuronState(i, savedstate[i]);
        }
        Agent.landmarkSensor = savedLS;
        Agent.foodSensor = savedFS;
        Agent.Step(StepSize);

       pos2file << Agent.pos << " ";

        // Record neural state
        ofstream neural2file("neuralTrain"+std::to_string(env)+"_"+std::to_string(k)+".dat");
        for (int i = 1; i <= N; i++)
            neural2file << Agent.NervousSystem.NeuronOutput(i) << " ";
        neural2file << endl;

        // Set average distance to 0
        totaldist = 0.0;
        totaltime = 0.0;

        ref = 15.0;
        double sep = 15.0;

        // Run sim
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            // Sense
            Agent.Sense(ref,sep,LN,-1);

            // Move
            Agent.Step(StepSize);

            // Record position
            pos2file << Agent.pos << " ";
            // Record neural state
            for (int i = 1; i <= N; i++)
                neural2file << Agent.NervousSystem.NeuronOutput(i) << " ";
            neural2file << endl;

            // Measure distance between them (after transients)
            if (time > TransDuration)
            {
                loc = ref + env*sep;
                dist = fabs(Agent.pos - loc);
                totaldist += dist;
                totaltime += 1;
            }
        }
        neural2file.close();
        pos2file << endl;
        totalfit += 1 - ((totaldist / totaltime)/MinLength);
        perffile << env << "\t" << loc << "\t" << 1 - ((totaldist / totaltime)/MinLength) << endl;
        totaltrials += 1;
        k++;

        pos2file.close();
    }
    perffile << totalfit/totaltrials << endl;
    perffile.close();
}

// ------------------------------------
// Behavior
// ------------------------------------
void BehaviorGeneral(TVector<double> &genotype)
{

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    CountingAgent Agent(N);

    // Instantiate the nervous systems
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }

    // Save state
    TVector<double> savedstate;
    savedstate.SetBounds(1,N);
    double savedFS,savedLS;

    // Keep track of performance
    double dist, totaltime, totaldist, totaltrials = 0, totalfit = 0.0;
    double loc;

    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        ofstream perffile("perfgeneral"+std::to_string(env)+".dat");
        for (double ref = 12.0; ref <= 18.0; ref += 0.1)
        {
            for (double sep = 12.0; sep <= 18.0; sep += 0.1)
            {

                // Set average distance to 0
                totaldist = 0.0;
                totaltime = 0.0;

                // LEARNING PHASE
                // Reset agent's position and neural state
                Agent.ResetPosition(0);
                Agent.ResetNeuralState();

                // Run sim
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    // Sense
                    Agent.Sense(15,15,LN,env);
                    // Move
                    Agent.Step(StepSize);
                }

                // TESTING PHASE
                // ReSet agent's positions
                Agent.ResetPosition(0);

                // Run sim
                for (double time = 0; time < RunDuration; time += StepSize)
                {
                    // Sense
                    Agent.Sense(ref,sep,LN,-1);
                    // Move
                    Agent.Step(StepSize);

                    // Measure distance between them (after transients)
                    if (time > TransDuration)
                    {
                        loc = ref + env*sep;
                        dist = fabs(Agent.pos - loc);
                        totaldist += dist;
                        totaltime += 1;
                    }
                }
                perffile << ref << "\t" << sep << "\t" << 1 - ((totaldist / totaltime)/MinLength) << endl;
            }
        }
        perffile.close();
    }
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
    if (BestPerf > 0.99) {
        return 1;
    }
    else {
        return 0;
    }
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
    cout << Generation << " " << BestPerf << " " << AvgPerf << endl;
}

void ResultsDisplay(TSearch &s)
{
    TVector<double> bestVector;
    ofstream BestIndividualFile;
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);

    // Save the genotype of the best individual
    bestVector = s.BestIndividual();
    BestIndividualFile.open("best.gen.dat");
    BestIndividualFile << bestVector << endl;
    BestIndividualFile.close();

    // Also show the best individual in the Circuit Model form
    BestIndividualFile.open("best.ns.dat");
    GenPhenMapping(bestVector, phenotype);
    CountingAgent Agent(N);

    // Instantiate the nervous system
    Agent.NervousSystem.SetCircuitSize(N);
    int k = 1;
    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Bias
    for (int i = 1; i <= N; i++) {
        Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetFoodSensorWeight(i,phenotype(k));
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent.SetLandmarkSensorWeight(i,phenotype(k));
        k++;
    }
    BestIndividualFile << Agent.NervousSystem << endl;
    BestIndividualFile << Agent.foodsensorweights << "\n" << endl;
    BestIndividualFile << Agent.landmarksensorweights << "\n" << endl;
    BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[])
{
    long randomseed = static_cast<long>(time(NULL));
    if (argc == 2)
        randomseed += atoi(argv[1]);

    TSearch s(VectSize);

    #ifdef PRINTOFILE

    ofstream file;
    file.open("evol.dat");
    cout.rdbuf(file.rdbuf());

    // save the seed to a file
    ofstream seedfile;
    seedfile.open ("seed.dat");
    seedfile << randomseed << endl;
    seedfile.close();
    
    #endif
    
    // Configure the search
    s.SetRandomSeed(randomseed);
    s.SetSearchResultsDisplayFunction(ResultsDisplay);
    s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    s.SetSelectionMode(RANK_BASED);
    s.SetReproductionMode(GENETIC_ALGORITHM);
    s.SetPopulationSize(POPSIZE);
    s.SetMaxGenerations(GENS);
    s.SetCrossoverProbability(CROSSPROB);
    s.SetCrossoverMode(UNIFORM);
    s.SetMutationVariance(MUTVAR);
    s.SetMaxExpectedOffspring(EXPECTED);
    s.SetElitistFraction(ELITISM);
    s.SetSearchConstraint(1);

    /* Initialize and seed the search */
    s.InitializeSearch();
    
    //Evolve
    s.SetSearchTerminationFunction(TerminationFunction);
    s.SetEvaluationFunction(FitnessFunctionA);
    s.ExecuteSearch();

    s.SetSearchTerminationFunction(TerminationFunction);
    s.SetEvaluationFunction(FitnessFunctionB);
    s.ExecuteSearch();

    s.SetSearchTerminationFunction(TerminationFunction);
    s.SetEvaluationFunction(FitnessFunctionC);
    s.ExecuteSearch();    

    s.SetSearchTerminationFunction(NULL);
    s.SetEvaluationFunction(FitnessFunctionD);
    s.ExecuteSearch();

    // // Record behavior
    // if (s.BestPerformance() > 0.95){
    //     ifstream genefile;
    //     genefile.open("best.gen.dat");
    //     TVector<double> genotype(1, VectSize);
    //     genefile >> genotype;
    //     BehaviorSimple(genotype);
    // }

    #ifdef PRINTTOFILE
        evolfile.close();
    #endif

    return 0;
}
