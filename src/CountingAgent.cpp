// *******************************
// CTRNN
// *******************************

#include "CountingAgent.h"
#include "random.h"

// *******
// Control
// *******

// Init the agent
void CountingAgent::Set(int networksize)
{
	size = networksize;
	gain = 1.0; 
	foodsensorweights.SetBounds(1, size);
	foodsensorweights.FillContents(0.0);
	landmarksensorweights.SetBounds(1, size);
	landmarksensorweights.FillContents(0.0);
	pos = 0.0;
	foodSensor = 0.0;
	landmarkSensor = 0.0;
	NervousSystem.RandomizeCircuitState(0.0,0.0);
}

// Reset the state of the agent
void CountingAgent::ResetPosition(double initpos)
{
	pos = initpos;
}

// Reset the state of the agent
void CountingAgent::ResetNeuralState()
{
	NervousSystem.RandomizeCircuitState(0.0,0.0);
	landmarkSensor = 0.0;
	foodSensor = 0.0;
}

// Sense 
void CountingAgent::Sense(double ref, double sep, int numlandmarks, int foodpos)
{
	double loc, dist;
	for (int i = 1; i <= numlandmarks; i++){
		loc = ref + i*sep;
		dist = fabs(loc - pos);
		if (dist < 5)
		{
			landmarkSensor = 1/(1 + exp(8 * (dist - 1)));		
			if (i == foodpos)
			{
				foodSensor = landmarkSensor; 
			}
		}
	}
}

// Step
void CountingAgent::Step(double StepSize)
{
	// Set sensors to external input
	for (int i = 1; i <= size; i++){
		NervousSystem.SetNeuronExternalInput(i, foodSensor*foodsensorweights[i] + landmarkSensor*landmarksensorweights[i]);
	}

	// Update the nervous system
	NervousSystem.EulerStep(StepSize);

	// Update the body position
	pos += StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));
}
