package models.graph;


public class Vertex
{
	public String label; // label (e.g. "A")
	public boolean wasVisited;
	public Vertex (String lab) // constructor
	{
		label = lab;
		wasVisited = false;
	}
	public Vertex () // constructor
	{
		this("unindex node");
	}
	public void setVertexName (String lab) {
		label = lab;
	}
}
