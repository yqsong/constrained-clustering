package models.LDA;

import gnu.trove.TIntIntHashMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import models.clustering.MalletKmeansModify;


import cc.mallet.topics.LDAHyper;
import cc.mallet.cluster.Clusterer;
import cc.mallet.cluster.Clustering;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.NormalizedDotProductMetric;
import cc.mallet.util.Randoms;
import cc.mallet.util.Timing;

public class LDAHyperExtension extends LDAHyper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public LDAHyperExtension (int numberOfTopics) {
		this (numberOfTopics, numberOfTopics, DEFAULT_BETA);
	}
	
	public LDAHyperExtension (int numberOfTopics, double alphaSum, double beta) {
		this (numberOfTopics, alphaSum, beta, new Randoms());
	}
	
	public LDAHyperExtension(int numberOfTopics, double alphaSum, double beta, Randoms random) {
		super(numberOfTopics, alphaSum, beta, random);
	}
	
	private boolean isUseKmeans = false;
	
	public void isUseKmeansToInitialize(boolean isUse) {
		this.isUseKmeans = isUse;
	}
	
	@Override
	public void addInstances (InstanceList training) {
		initializeForTypes (training.getDataAlphabet());
		List<LabelSequence> topicSequences = new ArrayList<LabelSequence>();
	    //Modified By Yangqiu [Initialization by Clustering]
		int doc = 0;
		Clustering clusteringresults = null;
	    if (isUseKmeans == true) {
			Pipe instancePipe = new FeatureSequence2FeatureVector();
			InstanceList tfvectors = new InstanceList(instancePipe);			
			for(Instance inst : training){
				inst.unLock();
				Instance instnew = (Instance) inst.clone();
				instancePipe.pipe(instnew);
				tfvectors.add(instnew);
				inst.lock();
			}
			NormalizedDotProductMetric metric = new NormalizedDotProductMetric();
			Clusterer clusterer = new MalletKmeansModify(tfvectors.getPipe(), numTopics, metric, "maxmin");
			
			Timing timing = new Timing();
			clusteringresults = clusterer.cluster(tfvectors);
			timing.tick("Kmeans total:");
	    }
	    //Modified By Yangqiu End

	    System.err.println("document number: " + training.size());
		for (Instance instance : training) {
			doc++;
			LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[instanceLength(instance)]);
			if (false)
				// This method not yet obeying its last "false" argument, and must be for this to work
				sampleTopicsForOneDoc((FeatureSequence)instance.getData(), topicSequence, false, false);
			else {
				Randoms r = new Randoms();
				int[] topics = topicSequence.getFeatures();
				for (int i = 0; i < topics.length; i++) {
				    //Modified By Yangqiu [Initialization by Clustering]
				    if (isUseKmeans == true) {
				    	topics[i] = clusteringresults.getLabel(doc - 1);
				    } else {
				    	topics[i] = r.nextInt(numTopics);    	
				    }
				    //Modified By Yangqiu End
				}
					
			}
			topicSequences.add (topicSequence);
		}
		addInstances (training, topicSequences);
	}
	
	private void initializeForTypes (Alphabet alphabet) {
		if (this.alphabet == null) {
			this.alphabet = alphabet;
			this.numTypes = alphabet.size();
			this.typeTopicCounts = new TIntIntHashMap[numTypes];
			for (int fi = 0; fi < numTypes; fi++) 
				typeTopicCounts[fi] = new TIntIntHashMap();
			this.betaSum = beta * numTypes;
		} else if (alphabet != this.alphabet) {
			throw new IllegalArgumentException ("Cannot change Alphabet.");
		} else if (alphabet.size() != this.numTypes) {
			this.numTypes = alphabet.size();
			TIntIntHashMap[] newTypeTopicCounts = new TIntIntHashMap[numTypes];
			for (int i = 0; i < typeTopicCounts.length; i++)
				newTypeTopicCounts[i] = typeTopicCounts[i];
			for (int i = typeTopicCounts.length; i < numTypes; i++)
				newTypeTopicCounts[i] = new TIntIntHashMap();
			// TODO AKM July 18:  Why wasn't the next line there previously?
			// this.typeTopicCounts = newTypeTopicCounts;
			this.betaSum = beta * numTypes;
		}	// else, nothing changed, nothing to be done
	}
	
	public int getNumTypes() { return numTypes; }
	
	public double[] getAlpla() {return alpha;}

	public double getBeta() {return beta;}
	
	public double getBetaSum() {return betaSum;}
	
	public double[][] getDocumentTopicsData (boolean reform)
	{
		double[][] dataMatrix = new double[data.size()][];
		 // generate tables for annotation
		List<Double[]> docTopicDist = new ArrayList<Double[]>();
        int docLen = 0;
        double [] alpha = getAlpla();
        double alphaSum = 0.0;
		for (int i = 0; i < getNumTopics(); ++i) {
			alphaSum += alpha[i];
		}
		
		for (int i = 0; i < getData().size(); ++i) {
            double[] docTopic = new double[getNumTopics()];
			FeatureSequence fs = (FeatureSequence) getData().get(i).instance.getData();
			docLen = fs.getLength();

			LabelSequence topicSequence = getData().get(i).topicSequence;
			
			int[] currentDocTopics = topicSequence.getFeatures();//topic assignment

			Arrays.fill(docTopic, 0.0);
			for (int j = 0; j < docLen; ++j) {
				docTopic[currentDocTopics[j]]++;
			}			
			for (int j = 0; j < getNumTopics(); ++j) {
				if (docLen + alpha[j] != 0) {
					if (reform == false)
						docTopic[j] = ( docTopic[j] + alpha[j] ) / ( docLen + alphaSum );
					else {
						docTopic[j] = Math.sqrt( ( docTopic[j] + alpha[j] ) / ( docLen + alphaSum ) );
					}
				}			
			}
			dataMatrix[i] = docTopic;
        }
		return dataMatrix;
	}

}
