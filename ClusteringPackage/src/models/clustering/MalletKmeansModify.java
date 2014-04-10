package models.clustering;


/**
 * Modified By Yangqiu since the k-Means clustering in mallet does not work. 
 * Clusters a set of point via k-Means. The instances that are clustered are
 * expected to be of the type FeatureVector.
*/

import gnu.trove.TIntHashSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;
import java.util.Random;
import java.util.logging.Logger;

import cc.mallet.cluster.Clusterer;
import cc.mallet.cluster.Clustering;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2AugmentableFeatureVector;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Metric;
import cc.mallet.types.NormalizedDotProductMetric;
import cc.mallet.types.SparseVector;
import cc.mallet.util.Randoms;
import cc.mallet.util.Timing;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * KMeans Clustering
 * 
 * Clusters the points into k clusters by minimizing the total intra-cluster
 * variance. It uses a given {@link Metric} to find the distance between
 * {@link Instance}s, which should have {@link SparseVector}s in the data
 * field.
 * 
 */
public class MalletKmeansModify extends Clusterer {

	private Pipe instancePipe;

	private static final long serialVersionUID = 1L;

	// Stop after movement of means is less than this
	static double MEANS_TOLERANCE = 1e-2;

	// Maximum number of iterations
	static int MAX_ITER = 100;

	// Minimum fraction of points that move
	static double POINTS_TOLERANCE = .005;

	/**
	 * Treat an empty cluster as an error condition.
	 */
	public static final int EMPTY_ERROR = 0;
	/**
	 * Drop an empty cluster
	 */
	public static final int EMPTY_DROP = 1;
	/**
	 * Place the single instance furthest from the previous cluster mean
	 */
	public static final int EMPTY_SINGLE = 2;

	Random randinator;
	Metric metric;
	int numClusters;
	int emptyAction;
	List<SparseVector> clusterMeans;
	String initMethod;

	
	private static Logger logger = Logger
		.getLogger("com.ibm.clustering.newkmeans");

	/**
	 * Construct a KMeans object
	 * 
	 * @param instancePipe Pipe for the instances being clustered
	 * @param numClusters Number of clusters to use
	 * @param metric Metric object to measure instance distances
	 * @param emptyAction Specify what should happen when an empty cluster occurs
	 */
	public MalletKmeansModify(Pipe instancePipe, int numClusters, Metric metric,
			int emptyAction, String initMethod) {

		super(instancePipe);

		this.emptyAction = emptyAction;
		this.metric = metric;
		this.numClusters = numClusters;
		this.initMethod = initMethod;

		this.clusterMeans = new ArrayList<SparseVector>(numClusters);
		this.randinator = new Random();

	}

	/**
	 * Construct a KMeans object
	 * 
	 * @param instancePipe Pipe for the instances being clustered
	 * @param numClusters Number of clusters to use
	 * @param metric Metric object to measure instance distances <p/> If an empty
	 *        cluster occurs, it is considered an error.
	 */
	public MalletKmeansModify(Pipe instancePipe, int numClusters, Metric metric) {
		this(instancePipe, numClusters, metric, EMPTY_ERROR, "orthogonal");
	}
	
	public MalletKmeansModify(Pipe instancePipe, int numClusters, Metric metric, String initMethod) {
		this(instancePipe, numClusters, metric, EMPTY_ERROR, initMethod);
	}


	/**
	 * Cluster instances
	 * 
	 * @param instances List of instances to cluster
	 */
	@Override
	public Clustering cluster(InstanceList instances) {

		assert (instances.getPipe() == this.instancePipe);

		// Initialize clusterMeans
		Timing timing = new Timing();
		if(this.initMethod.equalsIgnoreCase("orthogonal")) {
			initializeOrthogonalMeansSample(instances, this.metric);
		} else if (initMethod.equalsIgnoreCase("maxmin")) {
			initializeMeansSample(instances, this.metric);
		}

		int clusterLabels[] = new int[instances.size()];
		List<InstanceList> instanceClusters = new ArrayList<InstanceList>(numClusters);
		int instClust;
		double instClustDist, instDist;
		double deltaMeans = Double.MAX_VALUE;
		double deltaPoints = instances.size();
		int iterations = 0;
		SparseVector clusterMean;

		for (int c = 0; c < numClusters; c++) {
			instanceClusters.add(c, new InstanceList(instancePipe));
		}

		logger.info("Entering KMeans iteration");
		timing.tick("Kmeans initialization:");

		while (deltaMeans > MEANS_TOLERANCE && iterations < MAX_ITER
				&& deltaPoints > instances.size() * POINTS_TOLERANCE) {

 			iterations++;
			deltaPoints = 0;

			// For each instance, measure its distance to the current cluster
			// means, and subsequently assign it to the closest cluster
			// by adding it to an corresponding instance list
			// The mean of each cluster InstanceList is then updated.
			for (int n = 0; n < instances.size(); n++) {

				instClust = 0;
				instClustDist = Double.MAX_VALUE;

				for (int c = 0; c < numClusters; c++) {
					instDist = metric.distance(clusterMeans.get(c),
							(SparseVector) instances.get(n).getData());

					if (instDist < instClustDist) {
						instClust = c;
						instClustDist = instDist;
					}
				}
				// Add to closest cluster & label it such
				instanceClusters.get(instClust).add(instances.get(n));

				if (clusterLabels[n] != instClust) {
					clusterLabels[n] = instClust;
					deltaPoints++;
				}

			}

			deltaMeans = 0;

			for (int c = 0; c < numClusters; c++) {

//				System.out.println("Cluster point number: " + instanceClusters.get(c).size());

				if (instanceClusters.get(c).size() > 0) {
					clusterMean = this.mean(instanceClusters.get(c));

					deltaMeans += metric.distance(clusterMeans.get(c), clusterMean);

					clusterMeans.set(c, clusterMean);

					instanceClusters.set(c, new InstanceList(instancePipe));

				} else {

					logger.info("Empty cluster found.");

					switch (emptyAction) {
					case EMPTY_ERROR:
						return null;
					case EMPTY_DROP:
						logger.fine("Removing cluster " + c);
						clusterMeans.remove(c);
						instanceClusters.remove(c);
						for (int n = 0; n < instances.size(); n++) {

							assert (clusterLabels[n] != c) : "Cluster size is "
								+ instanceClusters.get(c).size()
								+ "+ yet clusterLabels[n] is " + clusterLabels[n];

							if (clusterLabels[n] > c)
								clusterLabels[n]--;
						}

						numClusters--;
						c--; // <-- note this trickiness. bad style? maybe.
						// it just means now that we've deleted the entry,
						// we have to repeat the index to get the next entry.
						break;

					case EMPTY_SINGLE:

						// Get the instance the furthest from any centroid
						// and make it a new centroid.

						double newCentroidDist = 0;
						int newCentroid = 0;
						InstanceList cacheList = null;

						for (int clusters = 0; clusters < clusterMeans.size(); clusters++) {
							SparseVector centroid = clusterMeans.get(clusters);
							InstanceList centInstances = instanceClusters.get(clusters);

							// Dont't create new empty clusters.

							if (centInstances.size() <= 1)
								continue;
							for (int n = 0; n < centInstances.size(); n++) {
								double currentDist = metric.distance(centroid,
										(SparseVector) centInstances.get(n).getData());
								if (currentDist > newCentroidDist) {
									newCentroid = n;
									newCentroidDist = currentDist;
									cacheList = centInstances;

								}
							}
						}
						if (cacheList == null) {
							logger.info("Can't find an instance to move.  Exiting.");
							// Can't find an instance to move.
							return null;
						} else clusterMeans.set(c, (SparseVector) cacheList.get(
								newCentroid).getData());

					default:
						return null;
					}
				}

			}

			logger.fine("Iter " + iterations + " deltaMeans = " + deltaMeans);
			timing.tick(" Interaion " + iterations + " deltaMeans = " + deltaMeans + ": ");

		}

		if (deltaMeans <= MEANS_TOLERANCE) {
			logger.info("KMeans converged with deltaMeans = " + deltaMeans);
			logger.info("Iteration number = " + iterations);
		}
		else if (iterations >= MAX_ITER)
			logger.info("Maximum number of iterations (" + MAX_ITER + ") reached.");
		else if (deltaPoints <= instances.size() * POINTS_TOLERANCE)
			logger.info("Minimum number of points (np*" + POINTS_TOLERANCE + "="
					+ (int) (instances.size() * POINTS_TOLERANCE)
					+ ") moved in last iteration. Saying converged.");

		return new Clustering(instances, numClusters, clusterLabels);

	}

 	/**
 	 * Uses a MAX-MIN heuristic to seed the initial cluster means..
 	 * 
 	 * @param instList List of instances.
 	 * @param metric Distance metric.
 	 */

	private void initializeMeansSample(InstanceList instList, Metric metric) {

		// InstanceList has no remove() and null instances aren't
		// parsed out by most Pipes, so we have to pre-process
		// here and possibly leave some instances without
		// cluster assignments.

		List<Instance> instances = new ArrayList<Instance>(instList.size());
		for (int i = 0; i < instList.size(); i++) {
			Instance ins = instList.get(i);
			SparseVector sparse = (SparseVector) ins.getData();
			if (sparse.numLocations() == 0) {
//				System.out.println("null point¡¡" + i);
				continue;
			}
			instances.add(ins);
		}

		// Add next center that has the MAX of the MIN of the distances from
		// each of the previous j-1 centers (idea from Andrew Moore tutorial,
		// not sure who came up with it originally)

		for (int i = 0; i < numClusters; i++) {
			double max = 0;
			int selected = 0;
			for (int k = 0; k < instances.size(); k++) {
				double min = Double.MAX_VALUE;
				Instance ins = instances.get(k);
				SparseVector inst = (SparseVector) ins.getData();
				for (int j = 0; j < clusterMeans.size(); j++) {
					SparseVector centerInst = clusterMeans.get(j);
					double dist = metric.distance(centerInst, inst);
					if (dist < min)
						min = dist;
				}
				if (min > max) {
					selected = k;
					max = min;
				}
			}
			Instance newCenter = instances.remove(selected);
			clusterMeans.add((SparseVector) newCenter.getData());
		}
	}

 	/**
 	 * Uses a Orthogonal heuristic to seed the initial cluster means..
 	 * 
 	 * @param instList List of instances.
 	 * @param metric Distance metric.
 	 */

	private void initializeOrthogonalMeansSample(InstanceList instList, Metric metric) {

		// InstanceList has no remove() and null instances aren't
		// parsed out by most Pipes, so we have to pre-process
		// here and possibly leave some instances without
		// cluster assignments.

		List<Instance> instances = new ArrayList<Instance>(instList.size());
		for (int i = 0; i < instList.size(); i++) {
			Instance ins = instList.get(i);
			SparseVector sparse = (SparseVector) ins.getData();
			if (sparse.numLocations() == 0) {
//				System.out.println("null point¡¡" + i);
				continue;
			}
			instances.add(ins);
		}
		List<Double> orthValue = new ArrayList<Double>(instances.size());
		for (int i = 0; i < instances.size(); ++i) {
			orthValue.add(0.0);
		}
		// Add next center that is orthogonal to each of the previous j-1 centers

		Randoms r = new Randoms();
		int index = r.nextInt(instances.size());
		index = 0;
		Instance newCenter = instances.remove(index);
		orthValue.remove(index);
		clusterMeans.add((SparseVector) newCenter.getData());
		for (int i = 1; i < numClusters; i++) {
			int selected = 0;
			double min = Double.MAX_VALUE;
			SparseVector centerInst = clusterMeans.get(i - 1);
			for (int k = 0; k < instances.size(); k++) {
				
				Instance ins = instances.get(k);
				SparseVector inst = (SparseVector) ins.getData();
				double dist = orthValue.get(k) + centerInst.dotProduct(inst)/(centerInst.twoNorm() * inst.twoNorm() + Double.MIN_VALUE);
				
				orthValue.set(k, dist);
				if (dist < min) {
					min = dist;
					selected = k;
				}
			}
			System.out.println("Initialize center " + i + " using data " + selected);
//			System.out.println(" Instance size = " + instances.size());
			newCenter = instances.remove(selected);
			orthValue.remove(selected);
			clusterMeans.add((SparseVector) newCenter.getData());
		}

	}
	/**
	 * Return the ArrayList of cluster means after a run of the algorithm.
	 * 
	 * @return An ArrayList of Instances.
	 */

	public List<SparseVector> getClusterMeans() {
		return this.clusterMeans;
	}
	
	// Copy from mallet and fixed some bugs
    public SparseVector mean (InstanceList instances )
    {

		if (instances==null || instances.size()==0)
		    return null;
	
		Iterator<Instance> instanceItr = instances.iterator();
		
		SparseVector v;
		Instance instance;
		int indices[];
		int maxSparseIndex=-1;
		int maxDenseIndex=-1;
	
		// First, we find the union of all the indices used in the instances
		TIntHashSet hIndices = new TIntHashSet(instances.getDataAlphabet().size());
	
		while (instanceItr.hasNext())
		{ 
		    instance = instanceItr.next();
		    v = (SparseVector)(instance.getData());
		    indices = v.getIndices ();
	
		    //Modified By Yangqiu [v.numLocations() != 0]
		    if (indices!=null && indices.length != 0)
			//if (indices!=null)
			//Modified By Yangqiu End
		    {
				hIndices.addAll (indices);
		
				if (indices[indices.length-1]>maxSparseIndex)
				    maxSparseIndex = indices[indices.length-1];
		    }
		    else // dense
			if (v.numLocations()>maxDenseIndex)
			    maxDenseIndex = v.numLocations()-1;
		}
	
		if (maxDenseIndex>-1) // dense vectors were present
		{
		    if (maxSparseIndex>maxDenseIndex) 
		    // sparse vectors were present and they had greater indices than
		    // the dense vectors
		    { 
			// therefore, we create sparse vectors and 
			// add all the dense indices
			 for (int i=0 ; i<=maxDenseIndex ; i++)
			     hIndices.add (i); 
		    }
		    else
			// sparse indices may have been present, but we don't care
			// since they never had indices that exceeded those of the 
			// dense vectors
		    {
		    	return mean(instances, maxDenseIndex+1);
		    }
		}
		
		// reaching this statement implies we can create a sparse vector
		return mean (instances, hIndices.toArray ());

    }
    
    public SparseVector mean ( InstanceList instances, int numIndices )
    {
		SparseVector mv = new SparseVector (new double[numIndices], false);	
		return mean (instances, mv);
    }
    
    public SparseVector mean ( InstanceList instances, int[] indices )
    {
		// Create the mean vector with the indices having all zeros, 
		// nothing copied, sorted, and no checks for duplicates.
		SparseVector mv = new SparseVector (indices, new double[indices.length], false, true, false);	
		return this.mean (instances, mv);
    }

    private SparseVector mean (InstanceList instances, SparseVector meanVector )
    {
		if (instances==null || instances.size()==0)
		    return null;
		
		Instance instance;
		SparseVector v;
	
		Iterator<Instance> instanceItr = instances.iterator();
		
		double factor = 1.0/instances.size();
	
		while (instanceItr.hasNext())
		{    
		    instance = instanceItr.next();
		    v = (SparseVector)(instance.getData());
		    
		    meanVector.plusEqualsSparse (v, factor);
		}
		
		return meanVector;
    }
    
	// mini test
	public static void main (String[] args) {

//		double[][] feature = {
//				  {0.05, 0.05, 0.05, 0, 0, 0},
//				  {0.05, 0.05, 0.05, 0, 0, 0},
//				  {0, 0, 0, 0.05, 0.05, 0.05},
//				  {0, 0, 0, 0.05, 0.05, 0.05},
//				  {0.04, 0.04, 0, 0.04, 0.04, 0.04},
//				  {0.04, 0.04, 0.04, 0, 0.04, 0.04},
//				  };	
		
		double[][] feature = {
				  {0.05, 0.05, 0.05, 0, 0, 0},
				  {0.05, 0.05, 0.05, 0, 0, 0},
				  {0.07, 0.03, 0.03, 0, 0, 0},
				  {0.07, 0.03, 0.03, 0.02, 0, 0},
				  {0, 0, 0.02, 0.04, 0.03, 0.05},
				  {0, 0, 0, 0.04, 0.03, 0.05},
				  {0, 0, 0, 0.05, 0.05, 0.05},
				  {0, 0, 0, 0.05, 0.05, 0.05},
				  };	
//		GeneralKmeans kmeans = new GeneralKmeans(feature, 2);
//		kmeans.estimate();
//		int[] label = kmeans.getLabels();;
//		for (int i = 0; i < label.length; ++i) {
//			System.out.print(label[i] + " ");
//		}

		
	}
}	
