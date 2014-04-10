package Evaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

import models.clustering.ConstraintInformationTheoreticCoClustering;
import models.clustering.ConstraintKmeans;
import models.clustering.ConstraintSemiNMF;
import models.clustering.ConstraintSemiTriNMF;
import models.clustering.GeneralKmeans;
import models.clustering.InformationTheoreticCoClustering;
import models.datastructure.ColtSparseVector;
import models.datastructure.IDReverseSorter;
import net.didion.jwnl.JWNL;
import net.didion.jwnl.JWNLException;


import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import rita.wordnet.RiWordnet;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
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
import cc.mallet.types.SparseVector;
import cc.mallet.util.Timing;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.map.AbstractIntDoubleMap;
import cern.colt.map.OpenIntDoubleHashMap;
import cern.colt.map.OpenIntIntHashMap;
import cern.colt.matrix.DoubleMatrix1D;


public class Test20NG {
	
	protected static String[] categories = {
			"alt.atheism",
			"comp.graphics",
			"comp.os.ms.windows.misc",
			"comp.sys.ibm.pc.hardware",
			"comp.sys.mac.hardware",
			"comp.windows.x",
			"misc.forsale",
			"rec.autos",
			"rec.motorcycles",
			"rec.sport.baseball",
			"rec.sport.hockey",
			"sci.crypt",
			"sci.electronics",
			"sci.med",
			"sci.space",
			"soc.religion.christian",
			"talk.politics.guns",
			"talk.politics.mideast",
			"talk.politics.misc",
			"talk.religion.misc"
	};
	
	protected static double minDFCount = 2;
	protected static double maxDFCount = 10000000;
	protected static int seed = 0;
	protected static boolean isUseTFIDF = false;
	protected boolean isIsingModel = false;
	
    protected List<DoubleMatrix1D> dataMat = null;
    protected List<DoubleMatrix1D> tfidfData = null;
    
    protected List<String> docStringsAll = null;
    protected List<String> docURIsAll = null;
    protected List<Integer> docLabelsAll = null;
    
    protected List<String> docStrings = null;
    protected List<String> docURIs = null;
    protected List<Integer> docLabels = null;
    
    protected Alphabet alphabet = null;
    protected OpenIntIntHashMap forwardHash = null;
    protected OpenIntIntHashMap backwardHash = null;
    protected int[][] mustLinks = null;
    protected int[][] cannotLinks = null;
    protected int[][] wordMustLinks = null;
    protected int[][] wordCannotLinks = null;
    protected double[] docMustWeight = null;
    protected double[] docCannotWeight = null;
    protected double[] wordMustWeight = null;
    protected double[] wordCannotWeight = null;
	
    protected double docNEMustWMean = 0.0;
    protected double docNEMustWStd = 0.0;
    protected double docNEMustCorrectPerc = 0.0;
    
    //for wordnet
    protected List<HashMap<Integer, Double>> wordNetDist = null;
    protected HashMap<String, Integer> word2Index = null;
    protected HashMap<Integer, String> index2Word = null;
    
    public void initializeData (String inputDirectory, String queryStr, int clusterNum, double dataPercentage, File stopwordFile, boolean isShuffle) {
    	loadDataString (inputDirectory, queryStr);
    	loadFeatureVector(dataPercentage, clusterNum, stopwordFile, isShuffle);
    }
	
	public List<String> loadDataString (String inputDirectory, String queryStr) {
        IndexSearcher searcher = null;
        
        try{
//            IndexReader.open(inputDirectory);
            searcher = new IndexSearcher(inputDirectory);
        } catch (Exception e) {
            e.printStackTrace();
        }        
        
		QueryParser queryParser = new QueryParser("newsgroup", new StandardAnalyzer());
		
        docStringsAll = new ArrayList<String>();
        docURIsAll = new ArrayList<String>();
        docLabelsAll = new ArrayList<Integer>();
        HashMap<String, Integer> labelHash = new HashMap<String, Integer>();
        int globalLabelID = 0;
        int docIndex = 0;
        
        try {
			Query query = queryParser.parse(queryStr);
			TopDocs hits = searcher.search(query, null, searcher.maxDoc());
			
			ScoreDoc[] docs = hits.scoreDocs;
			
			for (ScoreDoc doc : docs) {
				Document document = searcher.doc(doc.doc);

				String uri = document.get(Constants.URI_FIELD_NAME);
                docURIsAll.add(uri);
                
                String content = document.get(Constants.PLAIN_TEXT_FIELD_NAME)
                			   + document.get(Constants.TITLE) ;
                docStringsAll.add(content);

                String labelStr = document.get("newsgroup");
//                System.out.println(labelStr);
                int label = 0;
                if (labelHash.containsKey(labelStr)) {
                	label = labelHash.get(labelStr);
                } else {
                	label = globalLabelID;
                	labelHash.put(labelStr, globalLabelID);
                	globalLabelID++;
                }
                docLabelsAll.add(label);
                
                if (docIndex % 1000 == 0) {
                	System.out.println(">>>[LOG]: Loaded String " + docIndex + " documents.");
                }           
                docIndex++;
			}
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}

//        int docNum = reader.numDocs();
//        try {
//            for (int i = 0; i < docNum; ++i) {
//                if (!reader.isDeleted(i)) {
//                    Document doc = reader.document(i);
//                    docIDsAll.add(i);
//                    
//                    String uri = doc.get(Constants.URI_FIELD_NAME);
//                    docURIsAll.add(uri);
//                    
//                    String content = doc.get(Constants.PLAIN_TEXT_FIELD_NAME);
//                    docStringsAll.add(content);
//
//                    String labelStr = doc.get("newsgroup");
////                    System.out.println(labelStr);
//                    int label = 0;
//                    if (labelHash.containsKey(labelStr)) {
//                    	label = labelHash.get(labelStr);
//                    } else {
//                    	label = globalLabelID;
//                    	labelHash.put(labelStr, globalLabelID);
//                    	globalLabelID++;
//                    }
//                    docLabelsAll.add(label);
//                    
//                    if (i % 1000 == 0) {
//                    	System.out.println(">>>[LOG]: Loaded String " + i + " documents.");
//                    }                    
//                }
//            }
//        } 
//        catch  (Exception e) {
//        	e.printStackTrace();
//        }
        
        System.out.println(">>>[LOAD]: END");
        
        return docStringsAll;
	}
	
	private InstanceList loadInstanceSubsetList (double dataPercentage, File stopwordFile, boolean isShuffle) {
		Pipe instancePipe;
		instancePipe = new SerialPipes (new Pipe[] {
                new Target2Label(),
                new Input2CharSequence(),
                ((Pipe) new CharSequence2TokenSequence()),
                ((Pipe) new TokenSequenceLowercase()),
                ((stopwordFile == null) ? ((Pipe) new TokenSequenceRemoveStopwords(false, true)) : ((Pipe) new TokenSequenceRemoveStopwords(false, true).addStopWords(stopwordFile))),
                ((Pipe) new TokenSequence2FeatureSequence()),
                ((Pipe) new FeatureSequence2FeatureVector()),
            });
             
		InstanceList tflist = new InstanceList(instancePipe); 
        
		
		 List<Integer> permIndex = new ArrayList<Integer>();
		 for (int i = 0; i < docStringsAll.size(); ++i) {
			 permIndex.add(i);
		 }
		 if (isShuffle)
			 Collections.shuffle(permIndex);
		 int subsetNum = (int)(docStringsAll.size() * dataPercentage);
		 List<Integer> subIndices = permIndex.subList(0, subsetNum);
		 docStrings = new ArrayList<String>();
		 docURIs = new ArrayList<String>();
		 docLabels = new ArrayList<Integer>();
		 for (int i = 0; i < subIndices.size(); ++i) {
			 String text = docStringsAll.get(subIndices.get(i));
			 Instance carrier;
			 carrier = instancePipe.instanceFrom(new Instance (text, 0, null, null));
			 SparseVector sv = (SparseVector) carrier.getData();
			 int[] index = sv.getIndices();
			 double[] value = sv.getValues();
			 if (index.length > 5 && value.length > 5) {
				 if (carrier.getData() instanceof InstanceList) {
					 tflist = (InstanceList) carrier.getData();
				 }
				 else {
					 tflist.add (carrier);
				 }
				 
				 docStrings.add(docStringsAll.get(subIndices.get(i)));
				 docURIs.add(docURIsAll.get(subIndices.get(i)));
				 docLabels.add(docLabelsAll.get(subIndices.get(i)));
			 }
		 }
		 
		 return tflist;
	}
	
	public void loadFeatureVector(double dataPercentage, int clusterNum, File stopwordFile, boolean isShuffle) {
		
		InstanceList tflist = loadInstanceSubsetList (dataPercentage, stopwordFile, isShuffle);
        
        alphabet = tflist.getAlphabet();
        dataMat = new ArrayList<DoubleMatrix1D>(); 
        tfidfData = new ArrayList<DoubleMatrix1D>(); 
        
        // get word DF
        // get mutual information
        List<DoubleMatrix1D> labelVectors = new  ArrayList<DoubleMatrix1D>(); 
        for (int i = 0; i < clusterNum; ++i) {
        	DoubleMatrix1D vector = new ColtSparseVector(tflist.size());
        	labelVectors.add(vector);
        }
        for (int i = 0; i < docLabels.size(); ++i) {
        	labelVectors.get(docLabels.get(i)).set(i, 1);
        }
        List<DoubleMatrix1D> featureVectors = new ArrayList<DoubleMatrix1D>(); 
        for (int i = 0; i < alphabet.size(); ++i) {
        	DoubleMatrix1D vector = new ColtSparseVector(tflist.size());
        	featureVectors.add(vector);
        }
        double[] mutualInformation = new double[alphabet.size()];

        int[] alphabetCount = new int[alphabet.size()];
        double[] dfCount = new double[alphabet.size()];
        double[] tfCount = new double[alphabet.size()];
        int docCount = 0;
        for (Instance carrier : tflist) {
            if (docCount % 1000 == 0) {
            	System.out.println(">>>[LOG]: Sum feature DF " + docCount + " documents.");
            }       
            SparseVector sv = (SparseVector) carrier.getData();
            int[] index = sv.getIndices();
            double[] value = sv.getValues();
            
            for (int j = 0; j < index.length; ++j) {
            	if (value[j] > 0.0) {
            		alphabetCount[index[j]] += value[j];
            		dfCount[index[j]] += 1;
            		tfCount[index[j]] += value[j];
            		featureVectors.get(index[j]).set(docCount, 1);
            	}
            }
            docCount++;

        }
        for (int i = 0; i < dfCount.length; ++i) {
        	if (dfCount[i] > 0) {
        		dfCount[i] = Math.log(tflist.size()/dfCount[i]);
        	}
        }
        Arrays.fill(mutualInformation, 0.1);
//        System.out.println("Compute Mutual Information for Features...");
//        for (int i = 0; i < featureVectors.size(); ++i) {
//        	double sum = 0;
//        	for (int j = 0; j < labelVectors.size(); ++j) {
//        		DoubleMatrix1D v1 = featureVectors.get(i);
//        		DoubleMatrix1D v2 = labelVectors.get(j);
//        		sum += computeMutualInformation(v1, v2);
//        	}
//        	mutualInformation[i] = sum;
//        }
//        System.out.println("Compute Mutual Information Done.");
        double[] temp = Arrays.copyOf(mutualInformation, mutualInformation.length);
        Arrays.sort(temp);
        double miThreshold = 0;//temp[mutualInformation.length - 1000];
        
        double[] tempTFCount = Arrays.copyOf(tfCount, tfCount.length);
        Arrays.sort(tempTFCount);
//        minDFCount = tempTFCount[tempTFCount.length - 5000];//temp[mutualInformation.length - 5000];
        
        // remove unusual words
        forwardHash = new OpenIntIntHashMap();
        backwardHash = new OpenIntIntHashMap();
        int wordIndex = 0;
        for (int i = 0; i < alphabetCount.length; ++i) {
        	if (alphabetCount[i] > minDFCount 
        			&& alphabetCount[i] < maxDFCount 
        			&& mutualInformation[i] > miThreshold) {
        		forwardHash.put(i, wordIndex);
        		backwardHash.put(wordIndex, i);
        		wordIndex++;
        	}
//        	if (i%1000 == 0)
//        		System.out.println("   word: " + wordIndex);
        }

        int totalWord = 0;
		docCount = 0;
        for (Instance carrier : tflist) {
        	if (docCount % 1000 == 0) {
            	System.out.println(">>>[LOG]: Convert to COLT vector " + docCount + " documents.");
            }       
            docCount++;
            SparseVector sv = (SparseVector) carrier.getData();
            int[] index = sv.getIndices();
            double[] value = sv.getValues();
            double tfSum = 0;
            DoubleMatrix1D tfVector = new ColtSparseVector(wordIndex);
            DoubleMatrix1D tfidfVector = new ColtSparseVector(wordIndex);
            for (int j = 0; j < index.length; ++j) {
            	if (value[j] > 0.0 
            			&& alphabetCount[index[j]] > minDFCount 
            			&& alphabetCount[index[j]] < maxDFCount
            			&& mutualInformation[j] > miThreshold) {
            		tfVector.setQuick(forwardHash.get(index[j]), value[j]);
            		tfSum += value[j];
            	}
            }
            totalWord += tfSum;
            IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			tfVector.getNonZeros(indexList, valueList);
			double tempsum = 0;
            for (int j = 0; j < indexList.size(); ++j) {
//            	double tempvalue = (valueList.get(j)/tfSum) * dfCount[backwardHash.get(indexList.get(j))];
            	double tempvalue = (valueList.get(j)) * dfCount[backwardHash.get(indexList.get(j))];
            	tempsum += tempvalue * tempvalue;
            	tfidfVector.set(indexList.get(j), tempvalue);
            }
//            tempsum = Math.sqrt(tempsum);
//            indexList = new IntArrayList();
//			valueList = new DoubleArrayList();
//			tfidfVector.getNonZeros(indexList, valueList);
//			for (int j = 0; j < indexList.size(); ++j) {
//            	double tempvalue = (valueList.get(j)/tempsum);
//            	tfidfVector.set(indexList.get(j), tempvalue);
//            }
			tfVector.trimToSize();
			tfidfVector.trimToSize();
            dataMat.add(tfVector);
            tfidfData.add(tfidfVector);
//            carrier.unLock();
//            carrier.clearSource();
//            carrier.setData(null);
        }
        
        System.out.println(">>>[LOG]: Doc number: " + dataMat.size());
        System.out.println(">>>[LOG]: Word number: " + totalWord);
        System.out.println(">>>[LOG]: Vocabulary size: " + forwardHash.size());

	}
	
	public double computeMutualInformation(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		    assert(v1.size() == v2.size());

		    double h1 = 0.0;
		    double h2 = 0.0;
		    double mi = 0.0;

		    double a = 0; 
		    double b = 0;
		    double c = 0; 
		    double d = 0;
		    double n = 0;
		    for (int i = 0; i < v1.size(); ++i) {

		    	if (v1.getQuick(i) != 0.0 && v2.getQuick(i) != 0.0) { 
		    		a++; 
		    	} else if (v1.getQuick(i) == 0.0 && v2.getQuick(i) != 0.0) { 
		    		b++; 
		    	} else if (v1.getQuick(i) != 0.0 && v2.getQuick(i) == 0.0) { 
		    		c++; 
		    	} else if (v1.getQuick(i) == 0.0 && v2.getQuick(i) == 0.0) { 
		    		d++; 
		    	}
		    	n++;
		    }
		    
		    double temp1, temp2, temp3, temp4;
		    if (a != 0 && (a + b) * (a + c) != 0)
		    	temp1 = a / n * Math.log( a * n / (a + b) / (a + c) ) / Math.log(2);
		    else 
		    	temp1 = 0;
		    
		    if (b != 0 && (b + d) * (a + b) != 0)
		    	temp2 = b / n * Math.log( b * n / (b + d) / (a + b) ) / Math.log(2);
		    else 
		    	temp2 = 0;
		    
		    if (c != 0 && (c + d) * (a + c) != 0)
		    	temp3 = c / n * Math.log( c * n / (c + d) / (a + c) ) / Math.log(2);
		    else
		    	temp3 = 0;
		    
		    if (d != 0 && (b + d) * (c + d) != 0)
		    	temp4 = d / n * Math.log( d * n / (b + d) / (c + d) ) / Math.log(2);
		    else 
		    	temp4 = 0;
		    
		    mi = temp1 + temp2 + temp3 + temp4;
		    
		    boolean isNormalized = true;
		    if (isNormalized == true) {
		    	if ((a + c) != 0)
		    		temp1 = ((a + c) / n) * Math.log((a + c) / n) / Math.log(2);
		    	else 
		    		temp1 = 0;
		    	
		    	if ((b + d) != 0)
		    		temp2 = ((b + d) / n) * Math.log((b + d) / n) / Math.log(2);
		    	else 
		    		temp2 = 0;
		    	
		    	if ((a + b) != 0)
		    		temp3 = ((a + b) / n) * Math.log((a + b) / n) / Math.log(2);
		    	else
		    		temp3 = 0;
		    	
		    	if ((c + d) != 0)
		    		temp4 = ((c + d) / n) * Math.log((c + d) / n) / Math.log(2);
		    	else temp4 = 0;
		    	
		    	h1 = temp1 + temp2;
		    	h2 = temp3 + temp4;
		    	
		    	// since a + b + c + d = n, we can make sure h1*h2 != 0
		    	if (h1 * h2 == 0) 
		    		mi = 0;
		    	else
		    		mi /= Math.sqrt(h1 * h2);
		    }

		    return mi; 
		    
		}
	
	public int[][] inferenceMustConstraints (int[][] mustlinks) {
		int[][] newLinks = null;
		
		List<AbstractIntDoubleMap> mustLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
		for (int i = 0; i < dataMat.size(); ++i) {
			AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
			mustLinkConstrains.add(hashmap);
		}
		
		for (int i = 0; i < mustlinks.length; ++i) {
			int from = mustlinks[i][0];
			int to = mustlinks[i][1];
			double instDist = 1.0;
			
			if (mustLinkConstrains.get(from).containsKey(to) == false) {
				mustLinkConstrains.get(from).put(to, instDist);
			}
			if (mustLinkConstrains.get(to).containsKey(from) == false) {
				mustLinkConstrains.get(to).put(from, instDist);
			}
		}
		
		
		return newLinks;
	}
	
	public void generateConstraintsFromNE (String inputDirectory, String[] neNames, int maxConstNum, int minimalPair, int seed) {
        IndexReader reader = null;
        List<int[]> must = new ArrayList<int[]>();
        List<Double> weight = new ArrayList<Double>();
        
        List<Integer> permIndex = new ArrayList<Integer>();
        for (int i = 0; i < docLabels.size(); ++i) {
        	permIndex.add(i);
        }
        Random random = new Random(seed);
        Collections.shuffle(permIndex, random);
        
        int countPos = 0;
        int countNeg = 0;
        int countEqual = 0;
     	int countAll = 0;
     	double maxW = 0.0;
     	double sumW = 0.0;
        try{
        	reader = IndexReader.open(inputDirectory);
        
        	int relationNum = 0;
        	
        	for (int o = 1; o <= permIndex.size() - 1; ++o) {
        		for (int i = 0; i < permIndex.size() - o; ++i) {
        			int j = i + o;
//			for (int i = 0; i < permIndex.size(); ++i) {
//				for (int j = i + 1; j < permIndex.size(); ++j) {
//					System.out.println("    i: " + i + " j: " + j);
					if (relationNum % 50000 == 0) {
						System.out.println("    Processed relation number: " + relationNum);
					}
					relationNum++;
					String uriI = docURIs.get(permIndex.get(i));
					String uriJ = docURIs.get(permIndex.get(j));

					Term uriTermI = new Term(Constants.URI_FIELD_NAME, uriI);
	                int docFreqI = reader.docFreq(uriTermI);
	                Term uriTermJ = new Term(Constants.URI_FIELD_NAME, uriJ);
	                int docFreqJ = reader.docFreq(uriTermJ);
	                if (docFreqI != 0 && docFreqJ != 0) {
	                	TermDocs termDocs = reader.termDocs(uriTermI);
		                termDocs.next();
		                Document docI = reader.document(termDocs.doc());
						
						termDocs = reader.termDocs(uriTermJ);
		                termDocs.next();
		                Document docJ = reader.document(termDocs.doc());
						
		                int w = 0;
						for (int k = 0; k < neNames.length; ++k) {
							String name = neNames[k];
							String[] necollI = docI.getValues(name);
							String[] necollJ = docJ.getValues(name);
							for (int m = 0; m < necollI.length; ++m) {
								for (int n = 0; n < necollJ.length; ++n) {
									if (necollI[m].equalsIgnoreCase(necollJ[n])) {
										w++;
									}
								}
							}
						}
//						minimalPair = 1;
						if (w > minimalPair && must.size() <= maxConstNum) {
							if (must.size() % 200 == 0)
								System.out.println("    Must num: " + must.size());
							
							countAll++;
							if (docLabels.get(permIndex.get(i)) == docLabels.get(permIndex.get(j))) {
								
//								System.out.println("    Must from: " + permIndex.get(i) + " to: " + permIndex.get(j));
								
								countEqual++;
								if (docLabels.get(permIndex.get(i)) == 0) {
									countPos++;
								} else if (docLabels.get(permIndex.get(i)) == 1) {
									countNeg++;
								}
								
							}
							int[] ml = new int[2];
							ml[0] = permIndex.get(i);
							ml[1] = permIndex.get(j);
							must.add(ml);
//							weight.add(1.0);
							weight.add((double)w);
							if (w > maxW) {
								maxW = w;
							}
							sumW += w;
							
							
						} 
						else if (must.size() > maxConstNum) {
							break;
						}
	                }
				}
			}
        } catch (Exception e) {
            e.printStackTrace();
        }
//        int temp = must.size() + 0;
//        Collections.shuffle(permIndex, random);
//		 for (int k = 1; k <= permIndex.size() - 1; ++k) {
//			 for (int i = 0; i < permIndex.size() - k; ++i) {
//				 int j = i + k;
//				 if (docLabels.get(permIndex.get(i)) == docLabels.get(permIndex.get(j)) && must.size() <= temp) {
////					 double r = random.nextDouble();
////					 if (r < 0.97) {
//						 int[] m = new int[2];
//						 m[0] = permIndex.get(i);
//						 m[1] = permIndex.get(j);
//						 must.add(m);
////					 } else {
////						 int[] c = new int[2];
////						 c[0] = subsetIndex.get(i);
////						 c[1] = subsetIndex.get(j);
////						 cannot.add(c);
////					 }
//				 } 
//				 else if (must.size() > temp) {
//					 break;
//				 }
//			 }
//		 }
        
        docNEMustCorrectPerc = (double)countEqual/(double)countAll;
        docNEMustWMean = sumW/weight.size();
        docNEMustWStd = 0.0;
        for (int i = 0; i < weight.size(); ++i) {
        	docNEMustWStd += (weight.get(i) - docNEMustWMean) * (weight.get(i) - docNEMustWMean);
        }
        docNEMustWStd = Math.sqrt(docNEMustWStd/(weight.size() - 1));
        
        System.out.println("    Correct must link percentage: " + docNEMustCorrectPerc);
        System.out.println("    Must link number: " + countAll 
        		+ " mean w: " + docNEMustWMean
        		+ " std w: " + docNEMustWStd);
        System.out.println("    Positive: " + countPos
        		+ " Negative: " + countNeg);
		 	
        mustLinks = new int[must.size()][2]; 
        docMustWeight = new double[weight.size()];
//        docMustWeight = null;
        for (int i = 0; i < must.size(); ++i) {
        	for (int j = 0; j < 2; ++j) {
        		mustLinks[i][j] = must.get(i)[j];
        	}
        	docMustWeight[i] = (weight.get(i)/(sumW/weight.size())) * (1/Math.sqrt((double)docURIs.size() + Double.MIN_NORMAL));
//        	docMustWeight[i] = (weight.get(i)/(sumW/weight.size())) * (weight.get(i)/(sumW/weight.size())) * (1/((double)docURIs.size() + Double.MIN_NORMAL));
//			 docMustWeight[i] = (weight.get(i)) * (1/((double)docURIs.size() + Double.MIN_NORMAL));
        }
        cannotLinks = null;
	}
	
	// percentage is the percentage of labeled points
	public void generateConstraints (double percentage, int maxConstNum, long seed) {
		 List<Integer> permIndex = new ArrayList<Integer>();
		 for (int i = 0; i < docLabels.size(); ++i) {
			 permIndex.add(i);
		 }
		 Random random = new Random(seed);
		 Collections.shuffle(permIndex, random);
		 int labelNum = (int)(docLabels.size() * percentage);
		 List<Integer> subsetIndex = permIndex.subList(0, labelNum);
//		 List<Integer> labelSubset = new ArrayList<Integer>();
//		 for (int i = 0; i < labelIndex.size(); ++i) {
//			 labelSubset.add(docLabels.get(labelIndex.get(i)));
//		 }
		 List<int[]> must = new ArrayList<int[]>();
		 List<int[]> cannot = new ArrayList<int[]>();
//		 for (int i = 0; i < permIndex.size(); ++i) {
//			for (int j = i + 1; j < permIndex.size(); ++j) {
		 for (int k = 1; k <= subsetIndex.size() - 1; ++k) {
			 for (int i = 0; i < subsetIndex.size() - k; ++i) {
				 int j = i + k;
				 if (docLabels.get(subsetIndex.get(i)) == docLabels.get(subsetIndex.get(j)) && must.size() <= maxConstNum) {
//					 double r = random.nextDouble();
//					 if (r < 0.97) {
						 int[] m = new int[2];
						 m[0] = subsetIndex.get(i);
						 m[1] = subsetIndex.get(j);
						 must.add(m);
//					 } else {
//						 int[] c = new int[2];
//						 c[0] = subsetIndex.get(i);
//						 c[1] = subsetIndex.get(j);
//						 cannot.add(c);
//					 }
				 } 
				 else if (docLabels.get(subsetIndex.get(i)) != docLabels.get(subsetIndex.get(j)) && cannot.size() <= maxConstNum) {
//					 double r = random.nextDouble();
//					 if (r < 0.97) {
						 int[] c = new int[2];
						 c[0] = subsetIndex.get(i);
						 c[1] = subsetIndex.get(j);
						 cannot.add(c);
//					 } else {
//						 int[] m = new int[2];
//						 m[0] = subsetIndex.get(i);
//						 m[1] = subsetIndex.get(j);
//						 must.add(m);
//					 }
				 } 
				 else if (must.size() > maxConstNum && cannot.size() > maxConstNum) {
					 break;
				 }
			 }
		 }
		 
		 mustLinks = new int[must.size()][2];
		 cannotLinks = new int[cannot.size()][2];
		 for (int i = 0; i < must.size(); ++i) {
			 for (int j = 0; j < 2; ++j) {
				 mustLinks[i][j] = must.get(i)[j];
			 }
		 }
		 for (int i = 0; i < cannot.size(); ++i) {
			 for (int j = 0; j < 2; ++j) {
				 cannotLinks[i][j] = cannot.get(i)[j];
			 }
		 }
	}
	
	public void generateWordConstraints (int clusterNum, int maxConstNum, int seed) {

		// count each class word freq.
		IDReverseSorter[][] idsorter = new IDReverseSorter[clusterNum][dataMat.get(0).size() + 1];
		IDReverseSorter[] idsorterAll = new IDReverseSorter[dataMat.get(0).size() + 1];
		for (int i = 0; i < idsorter.length; ++i) {
			for (int j = 0; j < idsorter[i].length; ++j) {
				IDReverseSorter sorter = new IDReverseSorter(j, 0);
				idsorter[i][j] = sorter;
			}
		}
		for (int i = 0; i < idsorterAll.length; ++i) {
			IDReverseSorter sorter = new IDReverseSorter(i, 0);
			idsorterAll[i] = sorter;
		}
		try {
			for (int i = 0; i < dataMat.size(); ++i) {
	//			System.out.println("data : " + i);
				DoubleMatrix1D vector = this.dataMat.get(i); 
				int label = docLabels.get(i);
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				vector.getNonZeros(indexList, valueList);
				for (int k = 0; k < indexList.size(); ++k) {
					int index = indexList.get(k);
					double value = valueList.get(k);
	//				System.out.println("index : " + index);
					int id = idsorter[label][index].getID();
					if (index != id) {
						System.err.println("ID doesn't match!");
					}
					double idValue = idsorter[label][index].getValue();
					idsorter[label][index].set(id, idValue + value);
					idValue = idsorterAll[index].getValue();
					idsorterAll[index].set(id, idValue + value);
				}
			} 
//			for (int i = 0; i < idsorter.length; ++i) {
//				for (int j = 0; j < idsorter[i].length; ++j) {
//					int id = idsorter[i][j].getID();
//					double idValue = idsorter[i][j].getValue();
//					double idValueAll = idsorterAll[j].getValue();
//					if (idValueAll - idValue > 0) {
//						idsorter[i][j].set(id, 0);
//					}
////					idsorter[i][j].set(id, idValue / idValueAll);
//				}
//			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		for (int i = 0; i < idsorter.length; ++i) {
			Arrays.sort(idsorter[i]);
		}
		
		List<List<IDReverseSorter>> idsorter_unique = new ArrayList<List<IDReverseSorter>>();
		int clusterTopWordNum = 500;
		for (int i = 0; i < idsorter.length; ++i) {
			List<IDReverseSorter> idsorter_class = new ArrayList<IDReverseSorter>();
			for (int j = 0; j < idsorter[i].length; ++j) {
				if (idsorter[i][j].getValue() > 0.0) {
					idsorter_class.add(idsorter[i][j]);
				}
			}
			if (clusterTopWordNum > idsorter_class.size()) {
				clusterTopWordNum = idsorter_class.size();
			}
//			Collections.shuffle(idsorter_class);
			idsorter_unique.add(idsorter_class);
		}

		for (int i = 0; i < idsorter.length; ++i) {
			List<IDReverseSorter> idsorter_class = idsorter_unique.get(i).subList(0, clusterTopWordNum);
			Random random = new Random(seed);
			Collections.shuffle(idsorter_class, random);
			idsorter_unique.set(i, idsorter_class);
		}
		
		List<int[]> must = new ArrayList<int[]>();
		List<int[]> cannot = new ArrayList<int[]>();
		
//		 for (int k = 1; k < labelIndex.size() - 1; ++k) {
//			 for (int i = 0; i < labelIndex.size() - k; ++i) {
		for (int m = 1; m <= clusterTopWordNum - 1; ++m) {
			for (int i = 0; i < clusterTopWordNum - m; ++i) {
				for (int j = 0; j < clusterNum; ++j) {
					int k = i + m;
					if (must.size() <= maxConstNum && 
							idsorter_unique.get(j).get(i).getValue() > 0 && 
							idsorter_unique.get(j).get(k).getValue() > 0) {
						int[] ml = new int[2];
						ml[0] = idsorter_unique.get(j).get(i).getID();
						ml[1] = idsorter_unique.get(j).get(k).getID();
						must.add(ml);
					} else if (must.size() > maxConstNum) {
						 break;
					}
				}
			}
		}
		
//		for (int i = 0; i < clusterTopWordNum; ++i) {
//			for (int j = 0; j < clusterTopWordNum - i; ++j) {
//				for (int m = 1; m <= clusterNum - 1; ++m) {
//					for (int n = 0; n < clusterNum - m; ++n) {
//						int k = i + j; 
//						int l = n + m;
//						if (cannot.size() <= maxConstNum && 
//								idsorter_unique.get(n).get(j).getValue() > 0 && 
//								idsorter_unique.get(l).get(k).getValue() > 0) {
//							int[] cl = new int[2];
//							cl[0] = idsorter_unique.get(n).get(j).getID();
//							cl[1] = idsorter_unique.get(l).get(k).getID();
//							cannot.add(cl);
//						}
//						else if (cannot.size() > maxConstNum) {
//							 break;
//						}
//					}
//				}
//			}
//		}
		
		if (must.size() > 0) {
			wordMustLinks = new int[must.size()][2];
			for (int i = 0; i < must.size(); ++i) {
				 for (int j = 0; j < 2; ++j) {
					 wordMustLinks[i][j] = must.get(i)[j];
				 }
			 }
		}
		
		if (cannot.size() > 0) {
			wordCannotLinks = new int[cannot.size()][2];
			for (int i = 0; i < cannot.size(); ++i) {
				for (int j = 0; j < 2; ++j) {
					wordCannotLinks[i][j] = cannot.get(i)[j];
				}
			}
		}
	}

	public String getWordString (int index) {
		String word = (String) alphabet.lookupObject(backwardHash.get(index));
		return word;
	}
	
	public void loadWordNetDist(String inputFileName) {
		
		DoubleMatrix1D vector = this.dataMat.get(0);
		int vocabularySize = vector.size();
		this.word2Index = new HashMap<String, Integer>();
		this.index2Word = new HashMap<Integer, String>();
		this.wordNetDist = new ArrayList<HashMap<Integer, Double>>();
		for (int i = 0; i < vocabularySize; ++i) {
			HashMap<Integer, Double> hash = new HashMap<Integer, Double>();
			wordNetDist.add(hash);
		}
		
		File wordNetStat = new File(inputFileName);
		File[] pairValues = wordNetStat.listFiles();
		for (int i = 0; i < pairValues.length; ++i) {
			if (i % 100 == 0)
				System.out.println("Read " + i + " files...");
			
			File pvFile = pairValues[i];
			String name = pvFile.getName();
			String[] tokens = name.split("_");
			String word1 = tokens[0];
			String content = "";
			try {
				content = FileUtils.readFileToString(pvFile);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			tokens = content.split(",");
			String[] terms = null;
			terms = tokens[0].trim().split(":");
			if (terms.length == 3) {
				int index = Integer.parseInt(terms[0].trim());
				double dist = Double.parseDouble(terms[1].trim());
				String word2 = terms[2].trim();
				if (word1.equalsIgnoreCase(word2) == true) {
					if (this.word2Index.containsKey(word1) == false) {
						this.word2Index.put(word1, index);
						this.index2Word.put(index, word1);
					} else {
						int indextemp = word2Index.get(word1);
						if (index != indextemp) {
							System.err.println("Error: word index wrong!");
						}
					}
				}
				HashMap<Integer, Double> hash = this.wordNetDist.get(index);
				for (int j = 1; j < tokens.length; ++j) {
					terms = tokens[j].split(":");
					if (terms.length == 3) {
						index = Integer.parseInt(terms[0].trim());
						dist = Double.parseDouble(terms[1].trim());
						word2 = terms[2].trim();
						if (this.word2Index.containsKey(word2) == false) {
							this.word2Index.put(word2, index);
							this.index2Word.put(index, word2);
						} else {
							int indextemp = word2Index.get(word2);
							if (index != indextemp) {
								System.err.println("Error: word index wrong!");
							}
						}
						hash.put(index, dist);
					}
				}
			}
			
		}
	}
	
	public int generateWordNetConstraints (double thresholdMust, int maxConstNum, String inputFileName) {
		
		if (this.wordNetDist == null) {
			loadWordNetDist(inputFileName);
		}
		
		List<int[]> must = new ArrayList<int[]>();
		List<Double> weight = new ArrayList<Double>();
		DoubleMatrix1D vector = this.dataMat.get(0);
		int vocabularySize = vector.size();
		for (int i = 0; i < vocabularySize; ++i) {
			if (this.index2Word.containsKey(i)) {
			
				String word1 = getWordString(i);
				String word2 = this.index2Word.get(i);
				if (word1.equalsIgnoreCase(word2) == false) {
					System.err.println("Error: word index wrong!");
				}
				
				HashMap<Integer, Double> hash = this.wordNetDist.get(i);
				Set<Integer> keys = hash.keySet();
				Object[] keyArray = (Object[]) keys.toArray();
				for (int j = 0; j < keyArray.length; ++j) {
					double score = hash.get((Integer)keyArray[j]);
					if (score <= thresholdMust) {
						int[] ml = new int[2];
						ml[0] = i;
						ml[1] = (Integer)keyArray[j];
						must.add(ml);
						weight.add(score);
					}
				}
			}
		}
		
		if (must.size() > 0) {
			if (must.size() < maxConstNum) {
				maxConstNum = must.size();
			} 
			List<Integer> permIndex = new ArrayList<Integer>();
			 for (int i = 0; i < must.size(); ++i) {
				 permIndex.add(i);
			 }
			 Collections.shuffle(permIndex);
			 wordMustLinks = new int[maxConstNum][2];
			 wordMustWeight = new double[maxConstNum];
			 for (int i = 0; i < maxConstNum; ++i) {
				 for (int j = 0; j < 2; ++j) {
					 wordMustLinks[i][j] = must.get(permIndex.get(i))[j];
//					 wordMustWeight[i] = 1/(Math.sqrt(vocabularySize) + Double.MIN_NORMAL) * (1 - weight.get(permIndex.get(i)));
					 wordMustWeight[i] = 1/(Math.sqrt(maxConstNum) + Double.MIN_NORMAL);
				 }
			 }
		}
		this.wordNetDist = null;
		this.index2Word = null;
		this.word2Index = null;
		return wordMustLinks.length;

	}

	public void evaluateWordNetConstraints (double binSize, String outputFolder1, String outputFolder2) {
		
		
		File outputFolderFile = new File(outputFolder1);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		outputFolderFile = new File(outputFolder2);
		files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		int binCount = (int) (1/binSize);
		int[] histgram = new int[binCount + 1];
		Arrays.fill(histgram, 0);
		
		DoubleMatrix1D vector = this.dataMat.get(0);
		int vocabularySize = vector.size();
		
		FileInputStream propertyHomeStream;
		try {
			String propertyHome = "./config/file_properties.xml";
			propertyHomeStream = new FileInputStream(propertyHome);
			JWNL.initialize(propertyHomeStream);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (JWNLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String wordnetHome = "./wordnet2.0/";
		RiWordnet wordnet = new RiWordnet(null, wordnetHome);
//		wordnet.setWordnetHome(wordnetHome);
//		Locale.setDefault(Locale.GERMAN);
		Timing time = new Timing();
		time.tick("Start to compute word similarity");
		
		List<Integer> indexArray = new ArrayList<Integer>();
		for (int i = 0; i < vocabularySize; ++i) {
			String word = getWordString(i);
			String pos = wordnet.getBestPos(word);
			if (pos != null && pos.equalsIgnoreCase("n") == true) {
				indexArray.add(i);
			}
			if (i % 500 == 0) {
				long timeSec = time.elapsedTime();
				System.out.println("i = " + i + "; Time = " + timeSec/60000 + "min");
			}
		}
		System.out.println("With " + indexArray.size() + " noun words");
		
		try {
			for (int i = 0; i < indexArray.size() - 1; ++i) {
				
				String word1 = getWordString(indexArray.get(i));
//				String pos1 = wordnet.getBestPos(word1);
//				if (pos1 == null || pos1.equalsIgnoreCase("n") == false) 
//					continue;
				
				File outputFile = new File(outputFolder1 + word1 + "_pairValues.txt");
				FileWriter writer = new FileWriter(outputFile, true);
				writer.write(indexArray.get(i) + ":0.0:" + word1 + ", ");
				
				for (int j = i + 1; j < indexArray.size(); ++j) {
					
					String word2 = getWordString(indexArray.get(j));
//					String pos2 = wordnet.getBestPos(word2);
//					if (pos2 == null || pos2.equalsIgnoreCase("n") == false) 
//						continue;
					
					double score = 1.0;
//					if (pos1!= null && pos2 != null &&
//							pos1.equalsIgnoreCase("n") && 
//							pos2.equalsIgnoreCase("n") && 
//							pos1.equalsIgnoreCase(pos2) == true) {
//						score = wordnet.getDistance(word1, word2, pos1);
//					}
					score = wordnet.getDistance(word1, word2, "n");
					
					if (score < 1) {
						writer.write(indexArray.get(j) + ":" + score + ":" + word2 + ", ");
					}
					
//					String[] pos1 = wordnet.getPos(word1);
//					String[] pos2 = wordnet.getPos(word2);
//					List<String> pos = new ArrayList<String>();
//					for (int k = 0; k < pos1.length; ++k) {
//						pos.add(pos1[k]);
//					}
//					for (int k = 0; k < pos2.length; ++k) {
//						if (Arrays.binarySearch(pos1, pos2[k]) == -1) {
//							pos.add(pos2[k]);
//						}
//					}
//					double score = 0.0;
//					for (int k = 0; k < pos.size(); ++k) {
//						score += wordnet.getDistance(word1, word2, pos.get(k));
//					}
//					if (pos.size() > 0) 
//						score /= (pos.size());
//					else
//						score = 1.0;
					
					int binIndex = (int)(score/binSize);
					histgram[binIndex]++;
					
					if (j % 500 == 0 || score < 0.2) {
						long timeSec = time.elapsedTime();
						System.out.println("i = " + i + ", word1 = " + word1 + "; j = ," + j + " word2 = " + word2 + "; Time = " + timeSec/60000 + "min");
						System.out.println("Score = " + score);
					}
				}
				writer.write("\n\r");
				writer.close();
			}
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		

		File outputStatFile = new File(outputFolder2 + "stat.txt");
		try {
			FileWriter writer = new FileWriter(outputStatFile, true);
			
			for (int i = 0; i < histgram.length; ++i) {
				writer.write(i * binSize + "," + histgram[i] + ", \n\r");
			}
			
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public double testKmeans (int cNum, int seed) {
		if (dataMat == null) {
			return -1;
		}
		int[] labels = null;
		
		boolean isUseTFIDF_local = isUseTFIDF;

		GeneralKmeans clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = new GeneralKmeans(dataMat, cNum, "maxmin", seed);
		} else {
			clusterer = new GeneralKmeans(tfidfData, cNum, "maxmin", seed);
		}
//		clusterer.setDistType("Euclidean");
        clusterer.estimate(); 
        labels = clusterer.getLabels();
		
        int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testConstraintKmeans (int cNum, int seed, boolean isWeighted) {
		if (dataMat == null) {
			return -1;
		}
		
		boolean isUseTFIDF_local = isUseTFIDF;
		
		int[] labels = null;
		ConstraintKmeans clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = new ConstraintKmeans(dataMat, cNum, "maxmin", mustLinks, cannotLinks, seed);
		} else {
			clusterer = new ConstraintKmeans(tfidfData, cNum, "maxmin", mustLinks, cannotLinks, seed);
		}
		if (isWeighted == true) {
			if (docMustWeight != null) 
				clusterer.setMustWeight(docMustWeight);
			if (docCannotWeight != null) 
				clusterer.setCannotWeight(docCannotWeight);
		}
		clusterer.setIsingModel(isIsingModel);
        clusterer.estimate(); 
        labels = clusterer.getLabels();
		
        int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
        
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testSemiNMF (int cNum, int seed) {
		if (dataMat == null) {
			return -1;
		}
		
		boolean isUseTFIDF_local = isUseTFIDF;
		
		int[] labels = null;
		int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
        
        ConstraintSemiNMF clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = 
				new ConstraintSemiNMF(dataMat, cNum, seed);
		} else {
			clusterer = 
				new ConstraintSemiNMF(tfidfData, cNum, seed);
		}
		
//		clusterer.setRowClusterLabels(classes);
        clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testConstraintSemiNMF (int cNum, int seed, boolean isWeighted) {
		if (dataMat == null) {
			return -1;
		}
		
		boolean isUseTFIDF_local = isUseTFIDF;
		
		int[] labels = null;
		int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
        
        ConstraintSemiNMF clusterer = null;
		
		if (isUseTFIDF_local == false) {
			clusterer = 
				new ConstraintSemiNMF(dataMat, cNum, mustLinks, cannotLinks, seed);
		} else {
			clusterer = 
				new ConstraintSemiNMF(tfidfData, cNum, mustLinks, cannotLinks, seed);
		}
		
		if (isWeighted == true) {
			if (docMustWeight != null) 
				clusterer.setRowMustWeight(docMustWeight);
			if (docCannotWeight != null) 
				clusterer.setRowCannotWeight(docCannotWeight);
		}
//		clusterer.setRowClusterLabels(classes);
		clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
        
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testSemiTriNMF (int cNum, int wordNum, int seed) {
		if (dataMat == null) {
			return -1;
		}
		
		boolean isUseTFIDF_local = isUseTFIDF;
		
		int[] labels = null;
		int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
        
        ConstraintSemiTriNMF clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = 
				new ConstraintSemiTriNMF(dataMat, cNum, wordNum, seed);
		} else {
			clusterer = 
				new ConstraintSemiTriNMF(tfidfData, cNum, wordNum, seed);
		}
		
//		clusterer.setRowClusterLabels(classes);
        clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testConstraintSemiTriNMF (int cNum, int wordNum, int seed, boolean isWeighted) {
		if (dataMat == null) {
			return -1;
		}
		
		boolean isUseTFIDF_local = isUseTFIDF;
		
		int[] labels = null;
		int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
        
        ConstraintSemiTriNMF clusterer = null;
		
		if (isUseTFIDF_local == false) {
			clusterer = 
				new ConstraintSemiTriNMF(dataMat, cNum, wordNum, mustLinks, cannotLinks, wordMustLinks, wordCannotLinks, seed);
		} else {
			clusterer = 
				new ConstraintSemiTriNMF(tfidfData, cNum, wordNum, mustLinks, cannotLinks, wordMustLinks, wordCannotLinks, seed);
		}
		
		if (isWeighted == true) {
			if (docMustWeight != null) 
				clusterer.setRowMustWeight(docMustWeight);
			if (docCannotWeight != null) 
				clusterer.setRowCannotWeight(docCannotWeight);
			if (wordMustWeight != null) 
				clusterer.setColumnMustWeight(wordMustWeight);
			if (wordCannotWeight != null) 
				clusterer.setColumnCannotWeight(wordCannotWeight);
		}
//		clusterer.setRowClusterLabels(classes);
		clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
        
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testITCC (int cNum, int wordNum, int seed) {
		if (dataMat == null) {
			return -1;
		}
		int[] labels = null;
		
		boolean isUseTFIDF_local = false;
		
		InformationTheoreticCoClustering clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = 
	        	new InformationTheoreticCoClustering(dataMat, cNum, wordNum, seed);
		} else {
			clusterer = 
	        	new InformationTheoreticCoClustering(tfidfData, cNum, wordNum, seed);
		}
		
        clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
        int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public double testConstraintITCC (int cNum, int wordNum, int seed, boolean isWeighted) {
		if (dataMat == null) {
			return -1;
		}
		int[] labels = null;
		
		boolean isUseTFIDF_local = false;
		
		ConstraintInformationTheoreticCoClustering clusterer = null;
		if (isUseTFIDF_local == false) {
			clusterer = 
				new ConstraintInformationTheoreticCoClustering(dataMat, cNum, wordNum, mustLinks, cannotLinks, wordMustLinks, wordCannotLinks, seed);
		} else {
			clusterer = 
				new ConstraintInformationTheoreticCoClustering(tfidfData, cNum, wordNum, mustLinks, cannotLinks, wordMustLinks, wordCannotLinks, seed);
		}
//		ConstraintInformationTheoreticCoClustering clusterer = 
//        	new ConstraintInformationTheoreticCoClustering(dataMat, cNum, wordNum, mustLinks, cannotLinks, wordMustLinks, wordCannotLinks, seed);
		if (isWeighted == true) {
			if (docMustWeight != null) 
				clusterer.setRowMustWeight(docMustWeight);
			if (docCannotWeight != null) 
				clusterer.setRowCannotWeight(docCannotWeight);
			if (wordMustWeight != null) 
				clusterer.setColumnMustWeight(wordMustWeight);
			if (wordCannotWeight != null) 
				clusterer.setColumnCannotWeight(wordCannotWeight);
		}
		clusterer.setIsingModel(isIsingModel);
        clusterer.estimate(); 
        labels = clusterer.getRowClusterLabels();
		
        int[] classes = new int[docLabels.size()];
        for (int i = 0; i < docLabels.size(); ++i) {
        	classes[i] = docLabels.get(i);
        }
		return Evaluators.NormalizedMutualInfo(classes, labels);
	}
	
	public void testContraintEffect (int cNum, String outputFolder,
			int maxIter, int maxPerc, int intervelConstNum,
			int wordConstNum1, int wordConstNum2
			) {
		
		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
//		int maxIter = 5;
//		int maxPerc = 2;
//		int intervelConstNum = 500;
//		int wordConstNum1 = 50000;
//		int wordConstNum2 = 100000;
		
		double[][] nmiKmeans = new double[maxIter][maxPerc];
		double[][] nmiConstraintKmeans = new double[maxIter][maxPerc];
		double[][] nmiITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_1 = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_2 = new double[maxIter][maxPerc];
		
		double[][] nmiNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintNMF = new double[maxIter][maxPerc];
		double[][] nmiTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_1 = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_2 = new double[maxIter][maxPerc];

		
		FileWriter writerkm;
		FileWriter writeritcc;
		FileWriter writerckm;
		FileWriter writercitcc;
		FileWriter writercitcc_1;
		FileWriter writercitcc_2;
		
		FileWriter writernmf;
		FileWriter writercnmf;
		FileWriter writertrinmf;
		FileWriter writerctrinmf;
		FileWriter writerctrinmf_1;
		FileWriter writerctrinmf_2;
		try {
			for (int iter = 0; iter < maxIter; ++iter) {
				Calendar dateTime = Calendar.getInstance();
				int seed = (int)dateTime.getTimeInMillis();
//				int seed = iter;
				
				for (int i = 0; i < maxPerc; ++i) {
					generateConstraints(1, intervelConstNum * (i), seed);
					
					System.err.println("Iteration: " + iter + " percentage: " + i);

					System.out.println("\n\r");
					File outputFile = new File(outputFolder + "kmeans.txt");
					
					nmiKmeans[iter][i] = testKmeans (cNum, seed);
					outputFile = new File(outputFolder + "kmeans.txt");
					writerkm = new FileWriter(outputFile, true);
					writerkm.write(nmiKmeans[iter][i] + ", ");
					writerkm.close();
					System.out.println("\n\r");
					
					nmiConstraintKmeans[iter][i] = testConstraintKmeans (cNum, seed, false);
					outputFile = new File(outputFolder + "ckmeans.txt");
					writerckm = new FileWriter(outputFile, true);
					writerckm.write(nmiConstraintKmeans[iter][i] + ", ");
					writerckm.close();
					System.out.println("\n\r");
					
					
					
					nmiNMF[iter][i] = testSemiNMF (cNum, seed);
					outputFile = new File(outputFolder + "nmf.txt");
					writernmf = new FileWriter(outputFile, true);
					writernmf.write(nmiNMF[iter][i] + ", ");
					writernmf.close();
					System.out.println("\n\r");
					
					nmiConstraintNMF[iter][i] = testConstraintSemiNMF (cNum, seed, false);
					outputFile = new File(outputFolder + "cnmf.txt");
					writercnmf = new FileWriter(outputFile, true);
					writercnmf.write(nmiConstraintNMF[iter][i] + ", ");
					writercnmf.close();
					System.out.println("\n\r");
					
					
					
					nmiTriNMF[iter][i] = testSemiTriNMF (cNum, cNum*2, seed);
					outputFile = new File(outputFolder + "trinmf.txt");
					writertrinmf = new FileWriter(outputFile, true);
					writertrinmf.write(nmiTriNMF[iter][i] + ", ");
					writertrinmf.close();
					System.out.println("\n\r");
					
					nmiConstraintTriNMF[iter][i] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "ctrinmf.txt");
					writerctrinmf = new FileWriter(outputFile, true);
					writerctrinmf.write(nmiConstraintTriNMF[iter][i] + ", ");
					writerctrinmf.close();
					System.out.println("\n\r");
					
					
					
					nmiITCC[iter][i] = testITCC (cNum, cNum*2, seed);
					outputFile = new File(outputFolder + "itcc.txt");
					writeritcc = new FileWriter(outputFile, true);
					writeritcc.write(nmiITCC[iter][i] + ", ");
					writeritcc.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC[iter][i] = testConstraintITCC (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "citcc.txt");
					writercitcc = new FileWriter(outputFile, true);
					writercitcc.write(nmiConstraintITCC[iter][i] + ", ");
					writercitcc.close();
					System.out.println("\n\r");
					
					
					
					
					generateWordConstraints (cNum, wordConstNum1, seed);
					
					
					nmiConstraintTriNMF_1[iter][i] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "ctrinmf" + wordConstNum1 + ".txt");
					writerctrinmf_1 = new FileWriter(outputFile, true);
					writerctrinmf_1.write(nmiConstraintTriNMF_1[iter][i] + ", ");
					writerctrinmf_1.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_1[iter][i] = testConstraintITCC (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "citcc" + wordConstNum1 + ".txt");
					writercitcc_1 = new FileWriter(outputFile, true);
					writercitcc_1.write(nmiConstraintITCC_1[iter][i] + ", ");
					writercitcc_1.close();
					System.out.println("\n\r");

					
					
					generateWordConstraints (cNum, wordConstNum2, seed);
					
					nmiConstraintTriNMF_2[iter][i] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "ctrinmf" + wordConstNum2 + ".txt");
					writerctrinmf_2 = new FileWriter(outputFile, true);
					writerctrinmf_2.write(nmiConstraintTriNMF_2[iter][i] + ", ");
					writerctrinmf_2.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_2[iter][i] = testConstraintITCC (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "citcc" + wordConstNum2 + ".txt");
					writercitcc_2 = new FileWriter(outputFile, true);
					writercitcc_2.write(nmiConstraintITCC_2[iter][i] + ", ");
					writercitcc_2.close();
					System.out.println("\n\r");
					
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;

				}
				File outputFile = new File(outputFolder + "kmeans.txt");
				writerkm = new FileWriter(outputFile, true);
				writerkm.write("\n\r");
				writerkm.close();
				
				outputFile = new File(outputFolder + "ckmeans.txt");
				writerckm = new FileWriter(outputFile, true);
				writerckm.write("\n\r");
				writerckm.close();
				
				outputFile = new File(outputFolder + "nmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "cnmf.txt");
				writercnmf = new FileWriter(outputFile, true);
				writercnmf.write("\n\r");
				writercnmf.close();
				
				outputFile = new File(outputFolder + "trinmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "ctrinmf.txt");
				writercnmf = new FileWriter(outputFile, true);
				writercnmf.write("\n\r");
				writercnmf.close();
				
				outputFile = new File(outputFolder + "itcc.txt");
				writeritcc = new FileWriter(outputFile, true);
				writeritcc.write("\n\r");
				writeritcc.close();
				
				outputFile = new File(outputFolder + "citcc.txt");
				writercitcc = new FileWriter(outputFile, true);
				writercitcc.write("\n\r");
				writercitcc.close();

				
				
				outputFile = new File(outputFolder + "ctrinmf" + wordConstNum1 + ".txt");
				writercitcc_1 = new FileWriter(outputFile, true);
				writercitcc_1.write("\n\r");
				writercitcc_1.close();

				outputFile = new File(outputFolder + "ctrinmf" + wordConstNum2 + ".txt");
				writercitcc_2 = new FileWriter(outputFile, true);
				writercitcc_2.write("\n\r");
				writercitcc_2.close();
				
				outputFile = new File(outputFolder + "citcc" + wordConstNum1 + ".txt");
				writercitcc_1 = new FileWriter(outputFile, true);
				writercitcc_1.write("\n\r");
				writercitcc_1.close();

				outputFile = new File(outputFolder + "citcc" + wordConstNum2 + ".txt");
				writercitcc_2 = new FileWriter(outputFile, true);
				writercitcc_2.write("\n\r");
				writercitcc_2.close();

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		String strMean = "";
		String strStd = "";
		String strMax = "";
		strMean += "Constraints, ";
		strStd += "Constraints, ";
		strMax += "Constraints, ";
		for (int i = 0; i < maxPerc; ++i) {
			int number = intervelConstNum * (i);
			strMean += number + ", ";
			strStd += number + ", ";
			strMax += number + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		double[] mean = null; 
		double[] std = null;
		double[] max = null;
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiKmeans);
		strMean += "Kmeans, ";
		strStd += "Kmeans, ";
		strMax += "Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintKmeans);
		strMean += "Constraint Kmeans, ";
		strStd += "Constraint Kmeans, ";
		strMax += "Constraint Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		


		
		
		
		mean = ComputeMeanVariance.computeMean(nmiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiNMF);
		strMean += "Semi-NMF, ";
		strStd += "Semi-NMF, ";
		strMax += "Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintNMF);
		strMean += "Constraint Semi-NMF, ";
		strStd += "Constraint Semi-NMF, ";
		strMax += "Constraint Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiTriNMF);
		strMean += "Semi-Tri-NMF, ";
		strStd += "Semi-Tri-NMF, ";
		strMax += "Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF);
		strMean += "Constraint Semi-Tri-NMF, ";
		strStd += "Constraint Semi-Tri-NMF, ";
		strMax += "Constraint Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_1);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_1, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_1);
		strMean += "Constraint Semi-Tri-NMF (" + wordConstNum1 + "), ";
		strStd += "Constraint Semi-Tri-NMF (" + wordConstNum1 + "), ";
		strMax += "Constraint Semi-Tri-NMF (" + wordConstNum1 + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_2);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_2, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_2);
		strMean += "Constraint Semi-Tri-NMF (" + wordConstNum2 + "), ";
		strStd += "Constraint Semi-Tri-NMF (" + wordConstNum2 + "), ";
		strMax += "Constraint Semi-Tri-NMF (" + wordConstNum2 + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		

		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiITCC);
		strMean += "ITCC, ";
		strStd += "ITCC, ";
		strMax += "ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC);
		strMean += "Constraint ITCC, ";
		strStd += "Constraint ITCC, ";
		strMax += "Constraint ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_1);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_1, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_1);
		strMean += "Constraint ITCC (" + wordConstNum1 + "), ";
		strStd += "Constraint ITCC (" + wordConstNum1 + "), ";
		strMax += "Constraint ITCC (" + wordConstNum1 + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_2);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_2, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_2);
		strMean += "Constraint ITCC (" + wordConstNum2 + "), ";
		strStd += "Constraint ITCC (" + wordConstNum2 + "), ";
		strMax += "Constraint ITCC (" + wordConstNum2 + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax+= max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		
		FileWriter writer;
		File finalOutputFile = new File(outputFolder + "finalMean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalStd.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalMax.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMax);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void testDifferentNEConstraintEffect (int maxIter, int cNum, String inputNEDirectory, String outputFolder) {
		
		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		String[][] neStrs = {
				{"NE_LOCATION"},
				{"NE_PERSON"},
				{"NE_ORGANIZATION"},
				{"NE_LOCATION", "NE_PERSON"},
				{"NE_LOCATION", "NE_ORGANIZATION"},
				{"NE_PERSON", "NE_ORGANIZATION"},
				{"NE_LOCATION", "NE_PERSON", "NE_ORGANIZATION"},
		};
		String[] neStr = {
				"NE_LOCATION",
				"NE_PERSON",
				"NE_ORGANIZATION",
				"NE_LOCATION " + " NE_PERSON",
				"NE_LOCATION " + " NE_ORGANIZATION",
				"NE_PERSON " + " NE_ORGANIZATION",
				"NE_LOCATION " + " NE_PERSON "+ " NE_ORGANIZATION",
		};
		int[] neNums = {0, 1, 2, 3, 4, 5, 10};
		int maxPerc_1 = neStrs.length;
		int maxPerc_2 = neNums.length;
		
		double[][][] nmiKmeans = new double[maxIter][maxPerc_1][maxPerc_2];
		double[][][] nmiConstraintKmeans = new double[maxIter][maxPerc_1][maxPerc_2];
		
		double[][][] nmiSemiNMF = new double[maxIter][maxPerc_1][maxPerc_2];
		double[][][] nmiConstraintSemiNMF = new double[maxIter][maxPerc_1][maxPerc_2];
		
		double[][][] nmiSemiTriNMF = new double[maxIter][maxPerc_1][maxPerc_2];
		double[][][] nmiConstraintSemiTriNMF = new double[maxIter][maxPerc_1][maxPerc_2];		
		
		double[][][] nmiITCC = new double[maxIter][maxPerc_1][maxPerc_2];
		double[][][] nmiConstraintITCC = new double[maxIter][maxPerc_1][maxPerc_2];
		
		double[][] constraintNum = new double[maxPerc_1][maxPerc_2];
		double[][] constraintNumMean = new double[maxPerc_1][maxPerc_2];
		double[][] constraintNumStd = new double[maxPerc_1][maxPerc_2];
		double[][] constraintCorrectPerc = new double[maxPerc_1][maxPerc_2];
		
		Calendar dateTime = Calendar.getInstance();
		int seed1 = (int)dateTime.getTimeInMillis();
			for (int i = 0; i < maxPerc_1; ++i) {
				for (int j = 0; j < maxPerc_2; ++j) {
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;
					
					generateConstraintsFromNE (inputNEDirectory, neStrs[i], Integer.MAX_VALUE, neNums[j], seed);
					
					constraintNum[i][j] = this.mustLinks.length;
					constraintNumMean[i][j] = this.docNEMustWMean;
					constraintNumStd[i][j] = this.docNEMustWStd;
					constraintCorrectPerc[i][j] = this.docNEMustCorrectPerc;
					
					System.err.println("string: " + " i:" + neStr[i] + " num: " + neNums[j]);
					
					for (int k = 0; k < maxIter; ++k) {
						int seed = seed1 * (k + 1);
						
						System.err.println("Iteration: " + k);
						
						nmiKmeans[k][i][j] = testKmeans (cNum, seed);
						
						nmiConstraintKmeans[k][i][j] = testConstraintKmeans (cNum, seed, false);
						
						
						nmiSemiNMF[k][i][j] = testSemiNMF (cNum, seed);
						
						nmiConstraintSemiNMF[k][i][j] = testConstraintSemiNMF (cNum, seed, false);
						
						nmiSemiTriNMF[k][i][j] = testSemiTriNMF (cNum, cNum*2, seed);
						
						nmiConstraintSemiTriNMF[k][i][j] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);	
						
						
						nmiITCC[k][i][j] = testITCC (cNum, cNum*2, seed);
						
						nmiConstraintITCC[k][i][j] = testConstraintITCC (cNum, cNum*2, seed, false);
						
					}
				}
			}
		
		String strMean;
		String strStd;
		FileWriter writer;
		File finalOutputFile;
		double[][] mean = null; 
		double[][] std = null;
		
		mean = constraintNumMean;
		std = constraintNumStd;
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		String correctPercStr = " , ";
		String constraintNumStr = " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
			correctPercStr += neNums[i] + ", ";
			constraintNumStr += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		correctPercStr += "\n\r";
		constraintNumStr += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			correctPercStr += neStr[i] + ", ";
			constraintNumStr += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
				correctPercStr += constraintCorrectPerc[i][j] + ", ";
				constraintNumStr += constraintNum[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
			correctPercStr += "\n\r";
			constraintNumStr += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "constraint_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "constraint_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "constraint_perc.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(correctPercStr);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "constraint_num.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(constraintNumStr);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "kmeans_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "kmeans_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "ckmeans_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "ckmeans_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		
		
		
		
		
		
		mean = ComputeMeanVariance.computeMean(nmiSemiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiSemiNMF, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "nmf_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "nmf_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintSemiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintSemiNMF, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "cnmf_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "cnmf_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		mean = ComputeMeanVariance.computeMean(nmiSemiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiSemiTriNMF, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "trinmf_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "trinmf_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintSemiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintSemiTriNMF, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "ctrinmf_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "ctrinmf_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
		
		
		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "itcc_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "itcc_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC, mean);
		strMean = "";
		strStd = "";
		strMean += " , ";
		strStd += " , ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neNums[i] + ", ";
			strStd += neNums[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		for (int i = 0; i < mean.length; ++i) {
			strMean += neStr[i] + ", ";
			strStd += neStr[i] + ", ";
			for (int j = 0; j < mean[i].length; ++j) {
				strMean += mean[i][j] + ", ";
				strStd += std[i][j] + ", ";
			}
			strMean += "\n\r";
			strStd += "\n\r";
		}
		finalOutputFile = new File(outputFolder + "citcc_mean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "citcc_std.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public void testNEConstraintEffectforAllTwoClasses (int maxIter, String[] neNames, int minNE_Num, double dataPercentage, 
			String inputDirectory, File stopwordFile, String inputNEDirectory, String outputFolder) {
		
		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		int cNum = 2;		
		int maxPerc = 20 * 19 / 2;
		double[][] nmiKmeans = new double[maxIter][maxPerc];
		double[][] nmiConstraintKmeans = new double[maxIter][maxPerc];
		
		double[][] nmiSemiNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintSemiNMF = new double[maxIter][maxPerc];
		
		double[][] nmiSemiTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintSemiTriNMF = new double[maxIter][maxPerc];
		
		double[][] nmiITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC = new double[maxIter][maxPerc];
		
		double[] constraintNumMean = new double[maxPerc];
		double[] constraintNumStd = new double[maxPerc];
		double[] constraintCorrectPerc = new double[maxPerc];
		
		FileWriter writerkm;
		FileWriter writeritcc;
		FileWriter writerckm;
		FileWriter writercitcc;
		
		FileWriter writernmf;
		FileWriter writercnmf;
		FileWriter writertrinmf;
		FileWriter writerctrinmf;
		
		FileWriter writerConst;
		try {
			int percIndex = 0;
			File outputFile = new File(outputFolder + "temp");
			for (int i = 0; i < 19; ++i) {
				for (int j = i + 1; j < 20; ++j) {
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;
					
					String queryStr = "newsgroup:" + categories[i]
					       + " OR " + "newsgroup:" + categories[j];
					initializeData (inputDirectory, queryStr, cNum, dataPercentage, stopwordFile, true);
					generateConstraintsFromNE (inputNEDirectory, neNames, Integer.MAX_VALUE, minNE_Num, seed);
					
					constraintNumMean[percIndex] = this.docNEMustWMean;
					constraintNumStd[percIndex] = this.docNEMustWStd;
					constraintCorrectPerc[percIndex] = this.docNEMustCorrectPerc;
					
					outputFile = new File(outputFolder + "constraints.txt");
					writerConst = new FileWriter(outputFile, true);
					writerConst.write(this.docMustWeight.length +  ", "
							+ constraintNumMean[percIndex] + ", "
							+ constraintNumStd[percIndex] + ", "
							+ constraintCorrectPerc[percIndex] + ", " + "\n\r");
					writerConst.close();
					
					System.err.println("Classification: " + " i:" + i + " vs. " + " j:" + j);
					
					for (int k = 0; k < maxIter; ++k) {
						Calendar dateTime = Calendar.getInstance();
						int seed = (int)dateTime.getTimeInMillis();
//						int seed = iter;
						
						System.err.println("Iteration: " + k);

						System.out.println("\n\r");
						
						nmiKmeans[k][percIndex] = testKmeans (cNum, seed);
						outputFile = new File(outputFolder + "kmeans.txt");
						writerkm = new FileWriter(outputFile, true);
						writerkm.write(nmiKmeans[k][percIndex] + ", ");
						writerkm.close();
						System.out.println("\n\r");
						
						nmiConstraintKmeans[k][percIndex] = testConstraintKmeans (cNum, seed, false);
						outputFile = new File(outputFolder + "ckmeans.txt");
						writerckm = new FileWriter(outputFile, true);
						writerckm.write(nmiConstraintKmeans[k][percIndex] + ", ");
						writerckm.close();
						System.out.println("\n\r");
						
						
						
						
						nmiSemiNMF[k][percIndex] = testSemiNMF (cNum, seed);
						outputFile = new File(outputFolder + "nmf.txt");
						writernmf = new FileWriter(outputFile, true);
						writernmf.write(nmiSemiNMF[k][percIndex] + ", ");
						writernmf.close();
						System.out.println("\n\r");
						
						nmiConstraintSemiNMF[k][percIndex] = testConstraintSemiNMF (cNum, seed, false);
						outputFile = new File(outputFolder + "cnmf.txt");
						writercnmf = new FileWriter(outputFile, true);
						writercnmf.write(nmiConstraintSemiNMF[k][percIndex] + ", ");
						writercnmf.close();
						System.out.println("\n\r");
						
						
						nmiSemiTriNMF[k][percIndex] = testSemiTriNMF (cNum, cNum*2, seed);
						outputFile = new File(outputFolder + "trinmf.txt");
						writernmf = new FileWriter(outputFile, true);
						writernmf.write(nmiSemiTriNMF[k][percIndex] + ", ");
						writernmf.close();
						System.out.println("\n\r");
						
						nmiConstraintSemiTriNMF[k][percIndex] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);
						outputFile = new File(outputFolder + "ctrinmf.txt");
						writercnmf = new FileWriter(outputFile, true);
						writercnmf.write(nmiConstraintSemiTriNMF[k][percIndex] + ", ");
						writercnmf.close();
						System.out.println("\n\r");
						
						
						
						
						
						nmiITCC[k][percIndex] = testITCC (cNum, cNum*2, seed);
						outputFile = new File(outputFolder + "itcc.txt");
						writeritcc = new FileWriter(outputFile, true);
						writeritcc.write(nmiITCC[k][percIndex] + ", ");
						writeritcc.close();
						System.out.println("\n\r");
						
						nmiConstraintITCC[k][percIndex] = testConstraintITCC (cNum, cNum*2, seed, false);
						outputFile = new File(outputFolder + "citcc.txt");
						writercitcc = new FileWriter(outputFile, true);
						writercitcc.write(nmiConstraintITCC[k][percIndex] + ", ");
						writercitcc.close();
						System.out.println("\n\r");
						
					}
					percIndex++;
					
					outputFile = new File(outputFolder + "kmeans.txt");
					writerkm = new FileWriter(outputFile, true);
					writerkm.write("\n\r");
					writerkm.close();
					
					outputFile = new File(outputFolder + "itcc.txt");
					writerckm = new FileWriter(outputFile, true);
					writerckm.write("\n\r");
					writerckm.close();
					
					
					outputFile = new File(outputFolder + "nmf.txt");
					writernmf = new FileWriter(outputFile, true);
					writernmf.write("\n\r");
					writernmf.close();
					
					outputFile = new File(outputFolder + "cnmf.txt");
					writercnmf = new FileWriter(outputFile, true);
					writercnmf.write("\n\r");
					writercnmf.close();
					
					outputFile = new File(outputFolder + "trinmf.txt");
					writertrinmf = new FileWriter(outputFile, true);
					writertrinmf.write("\n\r");
					writertrinmf.close();
					
					outputFile = new File(outputFolder + "ctrinmf.txt");
					writerctrinmf = new FileWriter(outputFile, true);
					writerctrinmf.write("\n\r");
					writerctrinmf.close();
					
					
					outputFile = new File(outputFolder + "ckmeans.txt");
					writeritcc = new FileWriter(outputFile, true);
					writeritcc.write("\n\r");
					writeritcc.close();
					
					outputFile = new File(outputFolder + "citcc.txt");
					writercitcc = new FileWriter(outputFile, true);
					writercitcc.write("\n\r");
					writercitcc.close();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String strMean = "";
		String strStd = "";
		String strMax = "";
		strMean += "Classification, ";
		strStd += "Classification, ";
		strMax += "Classification, ";
		for (int i = 0; i < maxPerc; ++i) {
			int number = i;
			strMean += number + ", ";
			strStd += number + ", ";
			strMax += number + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		double[] mean = null; 
		double[] std = null;
		double[] max = null;
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiKmeans);
		strMean += "Kmeans, ";
		strStd += "Kmeans, ";
		strMax += "Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintKmeans);
		strMean += "Constraint Kmeans, ";
		strStd += "Constraint Kmeans, ";
		strMax += "Constraint Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		
		
		mean = ComputeMeanVariance.computeMean(nmiSemiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiSemiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiSemiNMF);
		strMean += "SemiNMF, ";
		strStd += "SemiNMF, ";
		strMax += "SemiNMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintSemiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintSemiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintSemiNMF);
		strMean += "Constraint SemiNMF, ";
		strStd += "Constraint SemiNMF, ";
		strMax += "Constraint SemiNMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiSemiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiSemiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiSemiTriNMF);
		strMean += "SemiTriNMF, ";
		strStd += "SemiTriNMF, ";
		strMax += "SemiTriNMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintSemiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintSemiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintSemiTriNMF);
		strMean += "Constraint SemiTriNMF, ";
		strStd += "Constraint SemiTriNMF, ";
		strMax += "Constraint SemiTriNMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		
		

		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiITCC);
		strMean += "ITCC, ";
		strStd += "ITCC, ";
		strMax+= "ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC);
		strMean += "Constraint ITCC, ";
		strStd += "Constraint ITCC, ";
		strMax += "Constraint ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax+= "\n\r";
		
		FileWriter writer;
		File finalOutputFile = new File(outputFolder + "finalMean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalStd.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalMax.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMax);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public void testWordClusterNumEffect (int cNum, String outputFolder,
			int maxIter, int maxPerc, int docConstNum,
			int wordConstNum) {

//		double[][] nmiKmeans = new double[maxIter][maxPerc];
//		double[][] nmiConstraintKmeans = new double[maxIter][maxPerc];
//		double[][] nmiITCC = new double[maxIter][maxPerc];
//		double[][] nmiConstraintITCC = new double[maxIter][maxPerc];
//		Calendar dateTime = Calendar.getInstance();
//		int seed = (int)dateTime.getTimeInMillis();
//		for (int iter = 0; iter < maxIter; ++iter) {
//			for (int i = 0; i < maxPerc; ++i) {
//				nmiKmeans[iter][i] = testKmeans (cNum, seed);
//				nmiConstraintKmeans[iter][i] = testITCC (cNum, (int)(cNum*Math.pow(2, i)), seed);
//				nmiITCC[iter][i] = testConstraintKmeans (cNum, seed);
//				nmiConstraintITCC[iter][i] = testConstraintITCC (cNum, (int)(cNum*Math.pow(2, i)), seed);
//			}
//		}
		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		double[][] nmiKmeans = new double[maxIter][maxPerc];
		double[][] nmiConstraintKmeans = new double[maxIter][maxPerc];
		double[][] nmiITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_1 = new double[maxIter][maxPerc];
		
		double[][] nmiNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintNMF = new double[maxIter][maxPerc];
		double[][] nmiTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_1 = new double[maxIter][maxPerc];

		
		FileWriter writerkm;
		FileWriter writeritcc;
		FileWriter writerckm;
		FileWriter writercitcc;
		FileWriter writercitcc_1;
		
		FileWriter writernmf;
		FileWriter writercnmf;
		FileWriter writertrinmf;
		FileWriter writerctrinmf;
		FileWriter writerctrinmf_1;
		try {
			for (int iter = 0; iter < maxIter; ++iter) {
				Calendar dateTime = Calendar.getInstance();
				int seed = (int)dateTime.getTimeInMillis();
//				int seed = iter;
				
				for (int i = 0; i < maxPerc; ++i) {
					
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;

					generateConstraints(1, docConstNum, seed);
					
					
					System.err.println("Iteration: " + iter + " percentage: " + i);

					System.out.println("\n\r");
					File outputFile = new File(outputFolder + "kmeans.txt");
					
					nmiKmeans[iter][i] = testKmeans (cNum, seed);
					outputFile = new File(outputFolder + "kmeans.txt");
					writerkm = new FileWriter(outputFile, true);
					writerkm.write(nmiKmeans[iter][i] + ", ");
					writerkm.close();
					System.out.println("\n\r");
					
					nmiConstraintKmeans[iter][i] = testConstraintKmeans (cNum, seed, false);
					outputFile = new File(outputFolder + "ckmeans.txt");
					writerckm = new FileWriter(outputFile, true);
					writerckm.write(nmiConstraintKmeans[iter][i] + ", ");
					writerckm.close();
					System.out.println("\n\r");
					
					
					
					nmiNMF[iter][i] = testSemiNMF (cNum, seed);
					outputFile = new File(outputFolder + "nmf.txt");
					writernmf = new FileWriter(outputFile, true);
					writernmf.write(nmiNMF[iter][i] + ", ");
					writernmf.close();
					System.out.println("\n\r");
					
					nmiConstraintNMF[iter][i] = testConstraintSemiNMF (cNum, seed, false);
					outputFile = new File(outputFolder + "cnmf.txt");
					writercnmf = new FileWriter(outputFile, true);
					writercnmf.write(nmiConstraintNMF[iter][i] + ", ");
					writercnmf.close();
					System.out.println("\n\r");
					
					
					
					nmiTriNMF[iter][i] = testSemiTriNMF (cNum, (int)(cNum*Math.pow(2, i)), seed);
					outputFile = new File(outputFolder + "trinmf.txt");
					writertrinmf = new FileWriter(outputFile, true);
					writertrinmf.write(nmiTriNMF[iter][i] + ", ");
					writertrinmf.close();
					System.out.println("\n\r");
					
					nmiConstraintTriNMF[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*Math.pow(2, i)), seed, false);
					outputFile = new File(outputFolder + "ctrinmf.txt");
					writerctrinmf = new FileWriter(outputFile, true);
					writerctrinmf.write(nmiConstraintTriNMF[iter][i] + ", ");
					writerctrinmf.close();
					System.out.println("\n\r");
					
					
					
					nmiITCC[iter][i] = testITCC (cNum, (int)(cNum*Math.pow(2, i)), seed);
					outputFile = new File(outputFolder + "itcc.txt");
					writeritcc = new FileWriter(outputFile, true);
					writeritcc.write(nmiITCC[iter][i] + ", ");
					writeritcc.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC[iter][i] = testConstraintITCC (cNum, (int)(cNum*Math.pow(2, i)), seed, false);
					outputFile = new File(outputFolder + "citcc.txt");
					writercitcc = new FileWriter(outputFile, true);
					writercitcc.write(nmiConstraintITCC[iter][i] + ", ");
					writercitcc.close();
					System.out.println("\n\r");
					
					
					generateWordConstraints (cNum, wordConstNum, seed);
					
					nmiConstraintTriNMF_1[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*Math.pow(2, i)), seed, false);
					outputFile = new File(outputFolder + "ctrinmf" + wordConstNum + ".txt");
					writerctrinmf_1 = new FileWriter(outputFile, true);
					writerctrinmf_1.write(nmiConstraintTriNMF_1[iter][i] + ", ");
					writerctrinmf_1.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_1[iter][i] = testConstraintITCC (cNum, (int)(cNum*Math.pow(2, i)), seed, false);
					outputFile = new File(outputFolder + "citcc" + wordConstNum + ".txt");
					writercitcc_1 = new FileWriter(outputFile, true);
					writercitcc_1.write(nmiConstraintITCC_1[iter][i] + ", ");
					writercitcc_1.close();
					System.out.println("\n\r");

					
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;

				}
				File outputFile = new File(outputFolder + "kmeans.txt");
				writerkm = new FileWriter(outputFile, true);
				writerkm.write("\n\r");
				writerkm.close();
				
				outputFile = new File(outputFolder + "ckmeans.txt");
				writerckm = new FileWriter(outputFile, true);
				writerckm.write("\n\r");
				writerckm.close();
				
				outputFile = new File(outputFolder + "nmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "cnmf.txt");
				writercnmf = new FileWriter(outputFile, true);
				writercnmf.write("\n\r");
				writercnmf.close();
				
				outputFile = new File(outputFolder + "trinmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "ctrinmf.txt");
				writercnmf = new FileWriter(outputFile, true);
				writercnmf.write("\n\r");
				writercnmf.close();
				
				outputFile = new File(outputFolder + "itcc.txt");
				writeritcc = new FileWriter(outputFile, true);
				writeritcc.write("\n\r");
				writeritcc.close();
				
				outputFile = new File(outputFolder + "citcc.txt");
				writercitcc = new FileWriter(outputFile, true);
				writercitcc.write("\n\r");
				writercitcc.close();

				
				
				outputFile = new File(outputFolder + "ctrinmf" + wordConstNum + ".txt");
				writercitcc_1 = new FileWriter(outputFile, true);
				writercitcc_1.write("\n\r");
				writercitcc_1.close();

				outputFile = new File(outputFolder + "citcc" + wordConstNum + ".txt");
				writercitcc_1 = new FileWriter(outputFile, true);
				writercitcc_1.write("\n\r");
				writercitcc_1.close();


			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		String strMean = "";
		String strStd = "";
		String strMax = "";
		strMean += "WordClusterNumber, ";
		strStd += "WordClusterNumber, ";
		strMax += "WordClusterNumber, ";
		for (int i = 0; i < maxPerc; ++i) {
			int number = (int)(cNum*Math.pow(2, i));
			strMean += number + ", ";
			strStd += number + ", ";
			strMax += number + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		double[] mean = null; 
		double[] std = null;
		double[] max = null;
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiKmeans);
		strMean += "Kmeans, ";
		strStd += "Kmeans, ";
		strMax += "Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintKmeans);
		strMean += "Constraint Kmeans, ";
		strStd += "Constraint Kmeans, ";
		strMax += "Constraint Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		


		
		
		
		mean = ComputeMeanVariance.computeMean(nmiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiNMF);
		strMean += "Semi-NMF, ";
		strStd += "Semi-NMF, ";
		strMax += "Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintNMF);
		strMean += "Constraint Semi-NMF, ";
		strStd += "Constraint Semi-NMF, ";
		strMax += "Constraint Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiTriNMF);
		strMean += "Semi-Tri-NMF, ";
		strStd += "Semi-Tri-NMF, ";
		strMax += "Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF);
		strMean += "Constraint Semi-Tri-NMF, ";
		strStd += "Constraint Semi-Tri-NMF, ";
		strMax += "Constraint Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_1);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_1, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_1);
		strMean += "Constraint Semi-Tri-NMF (" + wordConstNum + "), ";
		strStd += "Constraint Semi-Tri-NMF (" + wordConstNum + "), ";
		strMax += "Constraint Semi-Tri-NMF (" + wordConstNum + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		

		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiITCC);
		strMean += "ITCC, ";
		strStd += "ITCC, ";
		strMax += "ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC);
		strMean += "Constraint ITCC, ";
		strStd += "Constraint ITCC, ";
		strMax += "Constraint ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_1);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_1, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_1);
		strMean += "Constraint ITCC (" + wordConstNum + "), ";
		strStd += "Constraint ITCC (" + wordConstNum + "), ";
		strMax += "Constraint ITCC (" + wordConstNum + "), ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";


		
		FileWriter writer;
		File finalOutputFile = new File(outputFolder + "finalMean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalStd.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalMax.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMax);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void testWeightEffect (int cNum, String outputFolder,
			int maxIter, int maxPerc, int docConstNum,
			int wordConstNum) {

		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		double[][] nmiKmeans = new double[maxIter][maxPerc];
		double[][] nmiConstraintKmeans_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintKmeans_true = new double[maxIter][maxPerc];
		double[][] nmiITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_true = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_1_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC_1_true = new double[maxIter][maxPerc];
		
		double[][] nmiNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintNMF_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintNMF_true = new double[maxIter][maxPerc];
		double[][] nmiTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_true = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_1_false = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF_1_true = new double[maxIter][maxPerc];

		
		FileWriter writerkm;
		FileWriter writeritcc;
		FileWriter writerckm_false;
		FileWriter writerckm_true;
		FileWriter writercitcc_false;
		FileWriter writercitcc_true;
		FileWriter writercitcc_1_false;
		FileWriter writercitcc_1_true;
		
		FileWriter writernmf;
		FileWriter writercnmf_false;
		FileWriter writercnmf_true;
		FileWriter writertrinmf;
		FileWriter writerctrinmf_false;
		FileWriter writerctrinmf_true;
		FileWriter writerctrinmf_1_false;
		FileWriter writerctrinmf_1_true;
		try {
			for (int iter = 0; iter < maxIter; ++iter) {
				Calendar dateTime = Calendar.getInstance();
				int seed = (int)dateTime.getTimeInMillis();
//				int seed = iter;
				
				
				for (int i = 0; i < maxPerc; ++i) {
					
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;
					
					docMustWeight = null;
					docCannotWeight = null;
					wordMustWeight = null;
					wordCannotWeight = null;

					generateConstraints(1, docConstNum, seed);
					
					System.err.println("Iteration: " + iter + " percentage: " + i);

					System.out.println("\n\r");
					File outputFile = new File(outputFolder + "kmeans.txt");
					
					nmiKmeans[iter][i] = testKmeans (cNum, seed);
					outputFile = new File(outputFolder + "kmeans.txt");
					writerkm = new FileWriter(outputFile, true);
					writerkm.write(nmiKmeans[iter][i] + ", ");
					writerkm.close();
					System.out.println("\n\r");
					
					nmiConstraintKmeans_false[iter][i] = testConstraintKmeans (cNum, seed, false);
					outputFile = new File(outputFolder + "ckmeans_false.txt");
					writerckm_false = new FileWriter(outputFile, true);
					writerckm_false.write(nmiConstraintKmeans_false[iter][i] + ", ");
					writerckm_false.close();
					System.out.println("\n\r");
					
					docMustWeight = new double[mustLinks.length];
			        for (int j = 0; j < mustLinks.length; ++j) {
			        	docMustWeight[j] = Math.pow(10, (2 - i));
			        }
			        docCannotWeight = new double[cannotLinks.length];
			        for (int j = 0; j < cannotLinks.length; ++j) {
			        	docCannotWeight[j] = Math.pow(10, (2 - i));
			        }
			        
					nmiConstraintKmeans_true[iter][i] = testConstraintKmeans (cNum, seed, true);
					outputFile = new File(outputFolder + "ckmeans_true.txt");
					writerckm_true = new FileWriter(outputFile, true);
					writerckm_true.write(nmiConstraintKmeans_true[iter][i] + ", ");
					writerckm_true.close();
					System.out.println("\n\r");
					
					nmiNMF[iter][i] = testSemiNMF (cNum, seed);
					outputFile = new File(outputFolder + "nmf.txt");
					writernmf = new FileWriter(outputFile, true);
					writernmf.write(nmiNMF[iter][i] + ", ");
					writernmf.close();
					System.out.println("\n\r");
					
					nmiConstraintNMF_false[iter][i] = testConstraintSemiNMF (cNum, seed, false);
					outputFile = new File(outputFolder + "cnmf_false.txt");
					writercnmf_false = new FileWriter(outputFile, true);
					writercnmf_false.write(nmiConstraintNMF_false[iter][i] + ", ");
					writercnmf_false.close();
					System.out.println("\n\r");
					
					nmiConstraintNMF_true[iter][i] = testConstraintSemiNMF (cNum, seed, true);
					outputFile = new File(outputFolder + "cnmf_true.txt");
					writercnmf_true = new FileWriter(outputFile, true);
					writercnmf_true.write(nmiConstraintNMF_true[iter][i] + ", ");
					writercnmf_true.close();
					System.out.println("\n\r");
					
					
					nmiTriNMF[iter][i] = testSemiTriNMF (cNum, (int)(cNum*2), seed);
					outputFile = new File(outputFolder + "trinmf.txt");
					writertrinmf = new FileWriter(outputFile, true);
					writertrinmf.write(nmiTriNMF[iter][i] + ", ");
					writertrinmf.close();
					System.out.println("\n\r");
					
					nmiConstraintTriNMF_false[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*2), seed, false);
					outputFile = new File(outputFolder + "ctrinmf_false.txt");
					writerctrinmf_false = new FileWriter(outputFile, true);
					writerctrinmf_false.write(nmiConstraintTriNMF_false[iter][i] + ", ");
					writerctrinmf_false.close();
					System.out.println("\n\r");
					
					nmiConstraintTriNMF_true[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*2), seed, true);
					outputFile = new File(outputFolder + "ctrinmf_true.txt");
					writerctrinmf_true = new FileWriter(outputFile, true);
					writerctrinmf_true.write(nmiConstraintTriNMF_true[iter][i] + ", ");
					writerctrinmf_true.close();
					System.out.println("\n\r");
					
					
					
					nmiITCC[iter][i] = testITCC (cNum, (int)(cNum*2), seed);
					outputFile = new File(outputFolder + "itcc.txt");
					writeritcc = new FileWriter(outputFile, true);
					writeritcc.write(nmiITCC[iter][i] + ", ");
					writeritcc.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_false[iter][i] = testConstraintITCC (cNum, (int)(cNum*2), seed, false);
					outputFile = new File(outputFolder + "citcc_false.txt");
					writercitcc_false = new FileWriter(outputFile, true);
					writercitcc_false.write(nmiConstraintITCC_false[iter][i] + ", ");
					writercitcc_false.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_true[iter][i] = testConstraintITCC (cNum, (int)(cNum*2), seed, true);
					outputFile = new File(outputFolder + "citcc_true.txt");
					writercitcc_true = new FileWriter(outputFile, true);
					writercitcc_true.write(nmiConstraintITCC_true[iter][i] + ", ");
					writercitcc_true.close();
					System.out.println("\n\r");
					
					
					generateWordConstraints (cNum, wordConstNum, seed);
					
					wordMustWeight = new double[wordMustLinks.length];
			        for (int j = 0; j < wordMustLinks.length; ++j) {
			        	wordMustWeight[j] = Math.pow(10, (2 - i));
			        }
//			        wordCannotWeight = new double[wordCannotLinks.length];
//			        for (int j = 0; j < wordCannotLinks.length; ++j) {
//			        	wordCannotWeight[j] = Math.pow(10, (2 - i));
//			        }
			        
			        wordCannotWeight = null;
			        wordCannotLinks = null;
			        
			        mustLinks = null;
					cannotLinks = null;
			        docMustWeight = null;
			        docCannotWeight = null;
					
					nmiConstraintTriNMF_1_false[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*2), seed, false);
					outputFile = new File(outputFolder + "ctrinmf_false" + wordConstNum + ".txt");
					writerctrinmf_1_false = new FileWriter(outputFile, true);
					writerctrinmf_1_false.write(nmiConstraintTriNMF_1_false[iter][i] + ", ");
					writerctrinmf_1_false.close();
					System.out.println("\n\r");
					
					nmiConstraintTriNMF_1_true[iter][i] = testConstraintSemiTriNMF (cNum, (int)(cNum*2), seed, true);
					outputFile = new File(outputFolder + "ctrinmf_true" + wordConstNum + ".txt");
					writerctrinmf_1_true = new FileWriter(outputFile, true);
					writerctrinmf_1_true.write(nmiConstraintTriNMF_1_true[iter][i] + ", ");
					writerctrinmf_1_true.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_1_false[iter][i] = testConstraintITCC (cNum, (int)(cNum*2), seed, false);
					outputFile = new File(outputFolder + "citcc_false" + wordConstNum + ".txt");
					writercitcc_1_false = new FileWriter(outputFile, true);
					writercitcc_1_false.write(nmiConstraintITCC_1_false[iter][i] + ", ");
					writercitcc_1_false.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC_1_true[iter][i] = testConstraintITCC (cNum, (int)(cNum*2), seed, true);
					outputFile = new File(outputFolder + "citcc_true" + wordConstNum + ".txt");
					writercitcc_1_true = new FileWriter(outputFile, true);
					writercitcc_1_true.write(nmiConstraintITCC_1_true[iter][i] + ", ");
					writercitcc_1_true.close();
					System.out.println("\n\r");


				}
				File outputFile = new File(outputFolder + "kmeans.txt");
				writerkm = new FileWriter(outputFile, true);
				writerkm.write("\n\r");
				writerkm.close();
				
				outputFile = new File(outputFolder + "ckmeans_false.txt");
				writerckm_false = new FileWriter(outputFile, true);
				writerckm_false.write("\n\r");
				writerckm_false.close();

				outputFile = new File(outputFolder + "ckmeans_true.txt");
				writerckm_true = new FileWriter(outputFile, true);
				writerckm_true.write("\n\r");
				writerckm_true.close();

				
				outputFile = new File(outputFolder + "nmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "cnmf_false.txt");
				writercnmf_false = new FileWriter(outputFile, true);
				writercnmf_false.write("\n\r");
				writercnmf_false.close();

				outputFile = new File(outputFolder + "cnmf_true.txt");
				writercnmf_true = new FileWriter(outputFile, true);
				writercnmf_true.write("\n\r");
				writercnmf_true.close();

				
				outputFile = new File(outputFolder + "trinmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "ctrinmf_false.txt");
				writercnmf_false = new FileWriter(outputFile, true);
				writercnmf_false.write("\n\r");
				writercnmf_false.close();

				outputFile = new File(outputFolder + "ctrinmf_true.txt");
				writercnmf_true = new FileWriter(outputFile, true);
				writercnmf_true.write("\n\r");
				writercnmf_true.close();

				outputFile = new File(outputFolder + "itcc.txt");
				writeritcc = new FileWriter(outputFile, true);
				writeritcc.write("\n\r");
				writeritcc.close();
				
				outputFile = new File(outputFolder + "citcc_false.txt");
				writercitcc_false = new FileWriter(outputFile, true);
				writercitcc_false.write("\n\r");
				writercitcc_false.close();
				
				outputFile = new File(outputFolder + "citcc_true.txt");
				writercitcc_true = new FileWriter(outputFile, true);
				writercitcc_true.write("\n\r");
				writercitcc_true.close();

				
				
				outputFile = new File(outputFolder + "ctrinmf_false" + wordConstNum + ".txt");
				writercitcc_1_false = new FileWriter(outputFile, true);
				writercitcc_1_false.write("\n\r");
				writercitcc_1_false.close();

				outputFile = new File(outputFolder + "ctrinmf_true" + wordConstNum + ".txt");
				writercitcc_1_true = new FileWriter(outputFile, true);
				writercitcc_1_true.write("\n\r");
				writercitcc_1_true.close();
				
				outputFile = new File(outputFolder + "citcc_false" + wordConstNum + ".txt");
				writercitcc_1_false = new FileWriter(outputFile, true);
				writercitcc_1_false.write("\n\r");
				writercitcc_1_false.close();
				
				outputFile = new File(outputFolder + "citcc_true" + wordConstNum + ".txt");
				writercitcc_1_true = new FileWriter(outputFile, true);
				writercitcc_1_true.write("\n\r");
				writercitcc_1_true.close();


			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		String strMean = "";
		String strStd = "";
		String strMax = "";
		strMean += "WordClusterNumber, ";
		strStd += "WordClusterNumber, ";
		strMax += "WordClusterNumber, ";
		for (int i = 0; i < maxPerc; ++i) {
			double number = (double) Math.pow(10, (2 - i));
			strMean += number + ", ";
			strStd += number + ", ";
			strMax += number + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		double[] mean = null; 
		double[] std = null;
		double[] max = null;
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiKmeans);
		strMean += "Kmeans, ";
		strStd += "Kmeans, ";
		strMax += "Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintKmeans_false);
		strMean += "Constraint Kmeans Fixed, ";
		strStd += "Constraint Kmeans Fixed, ";
		strMax += "Constraint Kmeans Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintKmeans_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintKmeans_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintKmeans_true);
		strMean += "Constraint Kmeans Var, ";
		strStd += "Constraint Kmeans Var, ";
		strMax += "Constraint Kmeans Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		
		
		
		mean = ComputeMeanVariance.computeMean(nmiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiNMF);
		strMean += "Semi-NMF, ";
		strStd += "Semi-NMF, ";
		strMax += "Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintNMF_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintNMF_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintNMF_false);
		strMean += "Constraint Semi-NMF Fixed, ";
		strStd += "Constraint Semi-NMF Fixed, ";
		strMax += "Constraint Semi-NMF Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintNMF_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintNMF_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintNMF_true);
		strMean += "Constraint Semi-NMF Var, ";
		strStd += "Constraint Semi-NMF Var, ";
		strMax += "Constraint Semi-NMF Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiTriNMF);
		strMean += "Semi-Tri-NMF, ";
		strStd += "Semi-Tri-NMF, ";
		strMax += "Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_false);
		strMean += "Constraint Semi-Tri-NMF Fixed, ";
		strStd += "Constraint Semi-Tri-NMF Fixed, ";
		strMax += "Constraint Semi-Tri-NMF Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_true);
		strMean += "Constraint Semi-Tri-NMF Var, ";
		strStd += "Constraint Semi-Tri-NMF Var, ";
		strMax += "Constraint Semi-Tri-NMF Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_1_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_1_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_1_false);
		strMean += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Fixed, ";
		strStd += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Fixed, ";
		strMax += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF_1_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF_1_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF_1_true);
		strMean += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Var, ";
		strStd += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Var, ";
		strMax += "Constraint Semi-Tri-NMF (" + wordConstNum + ") Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		

		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiITCC);
		strMean += "ITCC, ";
		strStd += "ITCC, ";
		strMax += "ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_false);
		strMean += "Constraint ITCC Fixed, ";
		strStd += "Constraint ITCC Fixed, ";
		strMax += "Constraint ITCC Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_true);
		strMean += "Constraint ITCC Var, ";
		strStd += "Constraint ITCC Var, ";
		strMax += "Constraint ITCC Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_1_false);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_1_false, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_1_false);
		strMean += "Constraint ITCC (" + wordConstNum + ") Fixed, ";
		strStd += "Constraint ITCC (" + wordConstNum + ") Fixed, ";
		strMax += "Constraint ITCC (" + wordConstNum + ") Fixed, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC_1_true);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC_1_true, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC_1_true);
		strMean += "Constraint ITCC (" + wordConstNum + ") Var, ";
		strStd += "Constraint ITCC (" + wordConstNum + ") Var, ";
		strMax += "Constraint ITCC (" + wordConstNum + ") Var, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		
		FileWriter writer;
		File finalOutputFile = new File(outputFolder + "finalMean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalStd.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalMax.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMax);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void testWordNetConstraintEffect (int cNum, String inputFolder, String outputFolder,
			int maxIter, int maxPerc, int maxConstNum
			) {
		
		File outputFolderFile = new File(outputFolder);
		File[] files = outputFolderFile.listFiles();
		for (int i = 0; i < files.length; ++i) {
			files[i].delete();
		}
		
		double[][] nmiKmeans = new double[maxIter][maxPerc];
		double[][] nmiITCC = new double[maxIter][maxPerc];
		double[][] nmiConstraintITCC = new double[maxIter][maxPerc];
		
		double[][] nmiNMF = new double[maxIter][maxPerc];
		double[][] nmiTriNMF = new double[maxIter][maxPerc];
		double[][] nmiConstraintTriNMF = new double[maxIter][maxPerc];
		
		double[] threshold = new double[maxPerc];
		int[] constNum = new int[maxPerc];
		
		FileWriter writerkm;
		FileWriter writeritcc;
		FileWriter writercitcc;
		
		FileWriter writernmf;
		FileWriter writertrinmf;
		FileWriter writerctrinmf;
		try {
			
			for (int iter = 0; iter < maxIter; ++iter) {
				Calendar dateTime = Calendar.getInstance();
				int seed = (int)dateTime.getTimeInMillis();
//				int seed = iter;
				
				for (int i = 0; i < maxPerc; ++i) {
					
					System.err.println("Iteration: " + iter + " percentage: " + i);

					System.out.println("\n\r");
					File outputFile = new File(outputFolder + "kmeans.txt");
					
					nmiKmeans[iter][i] = testKmeans (cNum, seed);
					outputFile = new File(outputFolder + "kmeans.txt");
					writerkm = new FileWriter(outputFile, true);
					writerkm.write(nmiKmeans[iter][i] + ", ");
					writerkm.close();
					System.out.println("\n\r");
					
					nmiNMF[iter][i] = testSemiNMF (cNum, seed);
					outputFile = new File(outputFolder + "nmf.txt");
					writernmf = new FileWriter(outputFile, true);
					writernmf.write(nmiNMF[iter][i] + ", ");
					writernmf.close();
					System.out.println("\n\r");
//					
					nmiTriNMF[iter][i] = testSemiTriNMF (cNum, cNum*2, seed);
					outputFile = new File(outputFolder + "trinmf.txt");
					writertrinmf = new FileWriter(outputFile, true);
					writertrinmf.write(nmiTriNMF[iter][i] + ", ");
					writertrinmf.close();
					System.out.println("\n\r");
					
					nmiITCC[iter][i] = testITCC (cNum, cNum*2, seed);
					outputFile = new File(outputFolder + "itcc.txt");
					writeritcc = new FileWriter(outputFile, true);
					writeritcc.write(nmiITCC[iter][i] + ", ");
					writeritcc.close();
					System.out.println("\n\r");
					
					
					double thresholdMust = 0.05 + i * 0.05;
					threshold[i] = thresholdMust;
					constNum[i] = generateWordNetConstraints (thresholdMust, maxConstNum, inputFolder);
					
					nmiConstraintTriNMF[iter][i] = testConstraintSemiTriNMF (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "ctrinmf.txt");
					writerctrinmf = new FileWriter(outputFile, true);
					writerctrinmf.write(nmiConstraintTriNMF[iter][i] + ", ");
					writerctrinmf.close();
					System.out.println("\n\r");
					
					nmiConstraintITCC[iter][i] = testConstraintITCC (cNum, cNum*2, seed, false);
					outputFile = new File(outputFolder + "citcc.txt");
					writercitcc = new FileWriter(outputFile, true);
					writercitcc.write(nmiConstraintITCC[iter][i] + ", ");
					writercitcc.close();
					System.out.println("\n\r");
					
					
					mustLinks = null;
					cannotLinks = null;
					wordMustLinks = null;
					wordCannotLinks = null;
					wordMustWeight = null;

				}
				File outputFile = new File(outputFolder + "kmeans.txt");
				writerkm = new FileWriter(outputFile, true);
				writerkm.write("\n\r");
				writerkm.close();
				
				outputFile = new File(outputFolder + "nmf.txt");
				writernmf = new FileWriter(outputFile, true);
				writernmf.write("\n\r");
				writernmf.close();
				
				outputFile = new File(outputFolder + "trinmf.txt");
				writertrinmf = new FileWriter(outputFile, true);
				writertrinmf.write("\n\r");
				writertrinmf.close();
				
				outputFile = new File(outputFolder + "ctrinmf.txt");
				writerctrinmf = new FileWriter(outputFile, true);
				writerctrinmf.write("\n\r");
				writerctrinmf.close();
				
				outputFile = new File(outputFolder + "itcc.txt");
				writeritcc = new FileWriter(outputFile, true);
				writeritcc.write("\n\r");
				writeritcc.close();
				
				outputFile = new File(outputFolder + "citcc.txt");
				writercitcc = new FileWriter(outputFile, true);
				writercitcc.write("\n\r");
				writercitcc.close();

				

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String strMean = "";
		String strStd = "";
		String strMax = "";
		strMean += "Threshold, ";
		strStd += "Threshold, ";
		strMax += "Threshold, ";
		for (int i = 0; i < maxPerc; ++i) {
			double th = threshold[i];
			strMean += th + ", ";
			strStd += th + ", ";
			strMax += th + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		strMean += "Constraints, ";
		strStd += "Constraints, ";
		strMax += "Constraints, ";
		for (int i = 0; i < maxPerc; ++i) {
			int number = constNum[i];
			strMean += number + ", ";
			strStd += number + ", ";
			strMax += number + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		double[] mean = null; 
		double[] std = null;
		double[] max = null;
		mean = ComputeMeanVariance.computeMean(nmiKmeans);
		std = ComputeMeanVariance.computeStandardDeviation(nmiKmeans, mean);
		max = ComputeMeanVariance.computeMax(nmiKmeans);
		strMean += "Kmeans, ";
		strStd += "Kmeans, ";
		strMax += "Kmeans, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiNMF);
		strMean += "Semi-NMF, ";
		strStd += "Semi-NMF, ";
		strMax += "Semi-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiTriNMF);
		strMean += "Semi-Tri-NMF, ";
		strStd += "Semi-Tri-NMF, ";
		strMax += "Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiConstraintTriNMF);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintTriNMF, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintTriNMF);
		strMean += "Constraint Semi-Tri-NMF, ";
		strStd += "Constraint Semi-Tri-NMF, ";
		strMax += "Constraint Semi-Tri-NMF, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		mean = ComputeMeanVariance.computeMean(nmiITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiITCC);
		strMean += "ITCC, ";
		strStd += "ITCC, ";
		strMax += "ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";

		mean = ComputeMeanVariance.computeMean(nmiConstraintITCC);
		std = ComputeMeanVariance.computeStandardDeviation(nmiConstraintITCC, mean);
		max = ComputeMeanVariance.computeMax(nmiConstraintITCC);
		strMean += "Constraint ITCC, ";
		strStd += "Constraint ITCC, ";
		strMax += "Constraint ITCC, ";
		for (int i = 0; i < mean.length; ++i) {
			strMean += mean[i] + ", ";
			strStd += std[i] + ", ";
			strMax += max[i] + ", ";
		}
		strMean += "\n\r";
		strStd += "\n\r";
		strMax += "\n\r";
		
		FileWriter writer;
		File finalOutputFile = new File(outputFolder + "finalMean.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMean);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalStd.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strStd);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		finalOutputFile = new File(outputFolder + "finalMax.txt");
		try {
			writer = new FileWriter(finalOutputFile);
			writer.write(strMax);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] arg) {
		
	}
}
