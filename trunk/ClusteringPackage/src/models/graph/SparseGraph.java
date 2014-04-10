package models.graph;

import java.util.ArrayList;
import java.util.List;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.map.AbstractIntDoubleMap;
import cern.colt.map.OpenIntDoubleHashMap;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;

public class SparseGraph {
	/*	protected final int MAX_VERTS = 20;
	*/	protected Vertex vertexList[]; // vertex array
		protected List<AbstractIntDoubleMap> adjMat = null; // adjacency matrix
		protected int nVerts; // vertex number

		public SparseGraph(int vertexNum) 
		{
			nVerts = vertexNum;
			vertexList = new Vertex[vertexNum];
			for (int i = 0; i < vertexNum; ++i) {
				vertexList[i] = new Vertex(i+"");
			}
			adjMat = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < vertexNum; ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
				adjMat.add(hashmap);
			}
		} 

		public void addVertexName(int index, String lab)
		{
			vertexList[index].setVertexName(lab);
		}

		public void setEdge(int start, int end)
		{
			adjMat.get(start).put(end, 1);
			adjMat.get(end).put(start, 1);
		}
		
		public void setEdge(int start, int end, double value)
		{
			adjMat.get(start).put(end, value);
			adjMat.get(end).put(start, value);
		}

		public void displayVertex(int v)
		{
			System.out.print(vertexList[v].label);
		}

		// returns an unvisited vertex adjacent to v
		public int getAdjUnvisitedVertex(int v)
		{
			AbstractIntDoubleMap hash = adjMat.get(v);
			IntArrayList indexList = hash.keys();
			DoubleArrayList valueList = hash.values();
			for (int j = 0; j < indexList.size(); ++j){
				int index = indexList.get(j);
				double value = valueList.get(j);
				if (value > 0 && vertexList[index].wasVisited == false)
					return j;
			}
			return -1;
		}  
}
