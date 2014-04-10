package models.LSA;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;

import edu.ucla.sspace.common.ArgOptions;
import edu.ucla.sspace.mains.LoggerUtil;
import edu.ucla.sspace.text.IteratorFactory;

public class SSpaceLSAShell {
	protected LatentSemanticAnalysis space;
	
    public static final String CORPUS_READER_PROPERTY =
        "edu.ucla.sspace.mains.GenericMain.corpusReader";

    public static final String EXT = ".sspace";    

	protected void addExtraOptions(ArgOptions options) {
        options.addOption('n', "dimensions", 
                          "the number of dimensions in the semantic space",
                          true, "INT", "Algorithm Options"); 
        options.addOption('p', "preprocess", "a MatrixTransform class to "
                          + "use for preprocessing", true, "CLASSNAME",
                          "Algorithm Options");
        options.addOption('S', "svdAlgorithm", "a specific SVD algorithm to use"
                          , true, "SVD.Algorithm", 
                          "Advanced Algorithm Options");
    }
	
	protected String getAlgorithmSpecifics() {
        return 
            "The --svdAlgorithm provides a way to manually specify which " + 
            "algorithm should\nbe used internally.  This option should not be" +
            " used normally, as LSA will\nselect the fastest algorithm " +
            "available.  However, in the event that it\nis needed, valid" +
            " options are: SVDLIBC, SVDLIBJ, MATLAB, OCTAVE, JAMA and COLT\n";
    }
	
	protected void handleExtraOptions() { }
	
	protected Properties setupProperties() {
        Properties props = System.getProperties();
        return props;
    }

	protected void processDocumentsAndSpace(LatentSemanticAnalysis space,
			List<String> documents,
            int numThreads,
            Properties props) throws Exception {
		parseDocumentsSingleThreaded(space, documents);
		
		long startTime = System.currentTimeMillis();
		space.processSpace(props);
		long endTime = System.currentTimeMillis();
		System.out.println("processed space in " + ((endTime - startTime) / 1000d) + " seconds");
	}
	
	protected void parseDocumentsSingleThreaded(LatentSemanticAnalysis sspace,
			List<String> documents)
	throws IOException {
	
		long processStart = System.currentTimeMillis();
		int count = 0;
		
		for(int i = 0; i < documents.size(); ++i) {
			long startTime = System.currentTimeMillis();
			String doc = documents.get(i);
			int docNumber = ++count;
			sspace.processDocument(doc);
			long endTime = System.currentTimeMillis();
			System.out.println("processed document " +
			docNumber + " in " + ((endTime - startTime) / 1000d) + "seconds");
		}
		
		System.out.println("Processed all " + count + "documents in "
		+ ((System.currentTimeMillis() - processStart) / 1000d) + "  total seconds");            
	}
	
	public void run(List<String> documents, int dim) {
		
		LoggerUtil.setLevel(Level.FINE);

        // Check whether this class supports mutlithreading when deciding how
        // many threads to use by default
        int numThreads = 1;

        handleExtraOptions();

        Properties props = setupProperties();

        props.setProperty("edu.ucla.sspace.lsa.LatentSemanticAnalysis.dimensions", dim+"");
        
        
        IteratorFactory.setProperties(props);

		try {
			space = new LatentSemanticAnalysis();
			processDocumentsAndSpace(space, documents, numThreads, props);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public LatentSemanticAnalysis GetLSASenmanticAnalysis() {
		return this.space;
	}
    
}
