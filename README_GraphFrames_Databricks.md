# GraphFrames in Databricks

## Overview
This guide demonstrates how to use GraphFrames for distributed graph processing in Databricks. GraphFrames is a package for Apache Spark that provides DataFrame-based graphs, enabling scalable graph processing and analytics on large datasets.

## Why GraphFrames for Graph RAG?

GraphFrames offers several advantages for knowledge graph processing:
- **Scalability**: Built on Apache Spark, it handles large-scale graphs efficiently
- **Integration**: Seamless integration with Databricks and Spark SQL
- **Familiarity**: Uses DataFrame API, making it accessible to data engineers
- **Performance**: Distributed processing for complex graph algorithms
- **Flexibility**: Supports both graph queries and machine learning workflows

## Setup in Databricks

### 1. Install GraphFrames Package

In your Databricks notebook, install the GraphFrames library:

```python
# For Databricks Runtime 13.x or higher
%pip install graphframes
```

Or attach the GraphFrames library to your cluster:
- Go to your cluster configuration
- Navigate to Libraries → Install New
- Select Maven and use coordinates: `graphframes:graphframes:0.8.3-spark3.5-s_2.12`
  - Note: Adjust version based on your Spark version

### 2. Import Required Libraries

```python
from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, desc
import pyspark.sql.functions as F
```

## Example: Income Tax Returns Knowledge Graph

Building on the existing tax return data examples in this repository, here's how to create a knowledge graph using GraphFrames.

### 3. Create Vertices (Nodes)

```python
# Create vertices DataFrame - represents entities in the knowledge graph
# Example: States and tax return metrics

vertices = spark.createDataFrame([
    ("CA", "State", "California"),
    ("NY", "State", "New York"),
    ("TX", "State", "Texas"),
    ("FL", "State", "Florida"),
    ("returns_CA_1", "TaxReturns", "1234567"),
    ("returns_NY_1", "TaxReturns", "987654"),
    ("returns_TX_1", "TaxReturns", "1567890"),
    ("returns_FL_1", "TaxReturns", "1098765"),
    ("income_range_1", "IncomeRange", "$1 to $25,000"),
    ("income_range_2", "IncomeRange", "$25,000 to $50,000"),
    ("income_range_3", "IncomeRange", "$50,000 to $75,000"),
], ["id", "type", "name"])

vertices.display()
```

### 4. Create Edges (Relationships)

```python
# Create edges DataFrame - represents relationships between entities
edges = spark.createDataFrame([
    ("CA", "returns_CA_1", "HAS_RETURNS", "income_range_1"),
    ("NY", "returns_NY_1", "HAS_RETURNS", "income_range_2"),
    ("TX", "returns_TX_1", "HAS_RETURNS", "income_range_1"),
    ("FL", "returns_FL_1", "HAS_RETURNS", "income_range_3"),
    ("returns_CA_1", "income_range_1", "IN_RANGE", None),
    ("returns_NY_1", "income_range_2", "IN_RANGE", None),
    ("returns_TX_1", "income_range_1", "IN_RANGE", None),
    ("returns_FL_1", "income_range_3", "IN_RANGE", None),
], ["src", "dst", "relationship", "attribute"])

edges.display()
```

### 5. Create GraphFrame

```python
# Build the GraphFrame
g = GraphFrame(vertices, edges)

# Display basic graph statistics
print(f"Number of vertices: {g.vertices.count()}")
print(f"Number of edges: {g.edges.count()}")

# Show graph structure
print("\nVertices:")
g.vertices.show()

print("\nEdges:")
g.edges.show()
```

## Common Graph Operations

### Query by Vertex Type

```python
# Find all state nodes
states = g.vertices.filter(col("type") == "State")
states.display()

# Find all tax return nodes
tax_returns = g.vertices.filter(col("type") == "TaxReturns")
tax_returns.display()
```

### Query by Relationship

```python
# Find all HAS_RETURNS relationships
has_returns_edges = g.edges.filter(col("relationship") == "HAS_RETURNS")
has_returns_edges.display()

# Join to get state names with their tax returns
state_returns = (
    has_returns_edges
    .join(vertices.alias("state"), col("src") == col("state.id"))
    .join(vertices.alias("returns"), col("dst") == col("returns.id"))
    .select(
        col("state.name").alias("State"),
        col("returns.name").alias("Number_of_Returns"),
        col("attribute").alias("Income_Range")
    )
)
state_returns.display()
```

### Graph Algorithms

#### PageRank
Identify important nodes in the graph:

```python
# Run PageRank algorithm
results = g.pageRank(resetProbability=0.15, maxIter=10)

# Display vertices ranked by importance
results.vertices.select("id", "name", "pagerank")\
    .orderBy(desc("pagerank"))\
    .display()
```

#### Connected Components
Find connected subgraphs:

```python
# Find connected components
cc = g.connectedComponents()

# Display components
cc.select("id", "name", "component")\
    .orderBy("component")\
    .display()
```

#### Degree Analysis
Analyze node connectivity:

```python
# Calculate in-degree (incoming edges)
in_degrees = g.inDegrees
in_degrees.orderBy(desc("inDegree")).display()

# Calculate out-degree (outgoing edges)
out_degrees = g.outDegrees
out_degrees.orderBy(desc("outDegree")).display()

# Total degree
degrees = g.degrees
degrees.orderBy(desc("degree")).display()
```

### Graph Pattern Matching with Motif Finding

```python
# Find patterns: State -> Returns -> Income Range
motif = g.find("(state)-[e1]->(returns); (returns)-[e2]->(income)")

# Filter and display specific patterns
motif.filter("e1.relationship = 'HAS_RETURNS' AND e2.relationship = 'IN_RANGE'")\
    .select(
        "state.name",
        "returns.name",
        "income.name"
    ).display()
```

## Integration with RAG Pipeline

### 1. Build Knowledge Graph from Data

```python
# Load your tax return data
tax_data = spark.read.format("csv")\
    .option("header", "true")\
    .load("/path/to/tax_returns.csv")

# Transform data into graph format
# Create vertices from unique states and metrics
state_vertices = tax_data.select("STATE").distinct()\
    .withColumn("type", lit("State"))\
    .withColumn("name", col("STATE"))\
    .withColumnRenamed("STATE", "id")

# Create edges from relationships
edges_from_data = tax_data.select(
    col("STATE").alias("src"),
    col("ZIPCODE").alias("dst"),
    lit("LOCATED_IN").alias("relationship")
)

# Build GraphFrame
knowledge_graph = GraphFrame(state_vertices, edges_from_data)
```

### 2. Query Knowledge Graph for Context

```python
# Function to retrieve context for RAG
def get_graph_context(state_name, max_depth=2):
    """
    Retrieve relevant graph context for a given state
    to enhance LLM responses
    """
    # Find all connected entities within max_depth
    paths = g.bfs(
        fromExpr=f"name = '{state_name}'",
        toExpr="type = 'TaxReturns' OR type = 'IncomeRange'",
        maxPathLength=max_depth
    )
    
    return paths

# Example: Get context for California
ca_context = get_graph_context("California")
ca_context.display()
```

### 3. Export Graph for Vector Embeddings

```python
# Create text representations for embedding
def create_graph_text(graph):
    """
    Convert graph structure to text for LLM embedding
    """
    # Get all paths and relationships
    motifs = graph.find("(a)-[e]->(b)")
    
    # Create descriptive text
    text_data = motifs.select(
        F.concat_ws(" ",
            col("a.name"),
            col("e.relationship"),
            col("b.name")
        ).alias("text")
    )
    
    return text_data

# Generate text for embeddings
graph_texts = create_graph_text(g)
graph_texts.display()

# Save for embedding pipeline
graph_texts.write.format("delta").save("/path/to/graph_embeddings")
```

## Advanced Use Cases

### Subgraph Extraction

```python
# Extract subgraph for specific states
target_states = ["CA", "NY", "TX"]
subgraph_vertices = g.vertices.filter(col("id").isin(target_states))

# Get relevant edges
subgraph_edges = g.edges.filter(
    col("src").isin(target_states) | col("dst").isin(target_states)
)

# Create subgraph
subgraph = GraphFrame(subgraph_vertices, subgraph_edges)
```

### Graph Aggregations

```python
# Aggregate statistics by relationship type
relationship_stats = g.edges.groupBy("relationship")\
    .agg(count("*").alias("count"))\
    .orderBy(desc("count"))

relationship_stats.display()
```

### Time-Series Graph Analysis

```python
# For temporal graphs, add timestamp to edges
temporal_edges = edges.withColumn("timestamp", F.current_timestamp())

# Filter by time window
recent_edges = temporal_edges.filter(
    col("timestamp") > F.date_sub(F.current_date(), 30)
)

# Create temporal graph
temporal_graph = GraphFrame(vertices, recent_edges)
```

## Visualization

### Export for Visualization Tools

```python
# Export to format compatible with visualization tools
# Convert to NetworkX format for plotting
def export_to_networkx(graphframe):
    """
    Export GraphFrame to NetworkX for visualization
    """
    import networkx as nx
    
    # Collect vertices and edges
    vertices_pd = graphframe.vertices.toPandas()
    edges_pd = graphframe.edges.toPandas()
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for _, row in vertices_pd.iterrows():
        G.add_node(row['id'], type=row['type'], name=row['name'])
    
    # Add edges
    for _, row in edges_pd.iterrows():
        G.add_edge(row['src'], row['dst'], relationship=row['relationship'])
    
    return G

# Export and visualize
nx_graph = export_to_networkx(g)

# Use matplotlib or other visualization libraries
import matplotlib.pyplot as plt
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=8, arrows=True)
plt.savefig("graph_visualization.png")
display(plt.gcf())
```

### D3.js Export

```python
# Export to JSON for D3.js visualization
vertices_json = g.vertices.toJSON().collect()
edges_json = g.edges.toJSON().collect()

graph_json = {
    "nodes": vertices_json,
    "links": edges_json
}

import json
with open("/dbfs/FileStore/graph_data.json", "w") as f:
    json.dump(graph_json, f)
```

## Performance Optimization

### Caching

```python
# Cache frequently accessed graphs
g.vertices.cache()
g.edges.cache()

# Persist to disk if needed
g.vertices.write.format("delta").save("/path/to/cached_vertices")
g.edges.write.format("delta").save("/path/to/cached_edges")
```

### Partitioning

```python
# Repartition for better parallelism
vertices_partitioned = vertices.repartition(10, "type")
edges_partitioned = edges.repartition(20, "relationship")

g_optimized = GraphFrame(vertices_partitioned, edges_partitioned)
```

### Checkpoint

```python
# Set checkpoint directory for iterative algorithms
spark.sparkContext.setCheckpointDir("/dbfs/checkpoints")

# Checkpoint before expensive operations
g.vertices.checkpoint()
g.edges.checkpoint()
```

## Integration with Databricks Features

### Delta Lake Integration

```python
# Save graph as Delta tables for ACID transactions
vertices.write.format("delta").mode("overwrite")\
    .save("/mnt/delta/graph_vertices")

edges.write.format("delta").mode("overwrite")\
    .save("/mnt/delta/graph_edges")

# Read back from Delta
vertices_delta = spark.read.format("delta").load("/mnt/delta/graph_vertices")
edges_delta = spark.read.format("delta").load("/mnt/delta/graph_edges")

g_delta = GraphFrame(vertices_delta, edges_delta)
```

### MLflow Integration

```python
import mlflow

# Log graph metrics
with mlflow.start_run():
    mlflow.log_param("num_vertices", g.vertices.count())
    mlflow.log_param("num_edges", g.edges.count())
    
    # Log PageRank results
    pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
    avg_pagerank = pr_results.vertices.agg({"pagerank": "avg"}).collect()[0][0]
    mlflow.log_metric("avg_pagerank", avg_pagerank)
```

### Unity Catalog Integration

```python
# Save to Unity Catalog
vertices.write.format("delta")\
    .saveAsTable("catalog.schema.tax_graph_vertices")

edges.write.format("delta")\
    .saveAsTable("catalog.schema.tax_graph_edges")

# Read from Unity Catalog
vertices_uc = spark.table("catalog.schema.tax_graph_vertices")
edges_uc = spark.table("catalog.schema.tax_graph_edges")

g_uc = GraphFrame(vertices_uc, edges_uc)
```

## Comparison with Other Graph Approaches

| Feature | GraphFrames (Spark) | NetworkX | Neo4j |
|---------|-------------------|----------|-------|
| Scale | Distributed (TB+) | Single machine (GB) | Distributed (GB-TB) |
| Query Language | Spark SQL + Python | Python API | Cypher |
| Integration | Native Databricks | Python ecosystem | REST API/Bolt |
| Use Case | Big data analytics | Prototyping, small graphs | OLTP graph queries |
| Learning Curve | Medium (Spark knowledge) | Easy | Medium (Cypher) |

## Best Practices

1. **Data Modeling**
   - Keep vertex IDs unique and simple
   - Use meaningful relationship types
   - Add relevant attributes to both vertices and edges

2. **Performance**
   - Cache frequently accessed graphs
   - Partition large graphs appropriately
   - Use checkpointing for iterative algorithms
   - Leverage Delta Lake for efficient storage

3. **Graph Size**
   - For graphs < 1M nodes: NetworkX might be simpler
   - For graphs 1M-100M nodes: GraphFrames is ideal
   - For graphs > 100M nodes: Consider graph-specific databases

4. **RAG Integration**
   - Use graph queries to retrieve relevant context
   - Convert graph structure to text for embeddings
   - Combine graph context with vector search results
   - Update graphs incrementally as new data arrives

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**
   ```python
   # Increase executor memory in cluster configuration
   # Or partition your graph more aggressively
   vertices = vertices.repartition(100)
   ```

2. **Slow Graph Algorithms**
   ```python
   # Use sampling for large graphs
   sampled_edges = edges.sample(fraction=0.1)
   g_sample = GraphFrame(vertices, sampled_edges)
   ```

3. **GraphFrame Not Found**
   ```bash
   # Ensure correct version for your Spark version
   # Spark 3.5: graphframes:graphframes:0.8.3-spark3.5-s_2.12
   # Spark 3.4: graphframes:graphframes:0.8.2-spark3.4-s_2.12
   ```

## Resources

- [GraphFrames Documentation](https://graphframes.github.io/graphframes/docs/_site/index.html)
- [Databricks Graph Analysis Documentation](https://docs.databricks.com/en/index.html)
- [Apache Spark GraphX Guide](https://spark.apache.org/docs/latest/graphx-programming-guide.html)
- [Graph Algorithms with GraphFrames](https://databricks.com/blog/2016/03/03/introducing-graphframes.html)

## Examples in This Repository

- `Graph_RAG.ipynb` - NetworkX-based graph visualization
- `Graph_RAG_GraphDB.ipynb` - Neo4j graph database implementation
- This guide - Distributed graph processing with GraphFrames

## Next Steps

1. **Scale Your Graphs**: Start with small examples, then scale to production datasets
2. **Integrate with RAG**: Use graph queries to enhance retrieval for your LLM applications
3. **Explore Algorithms**: Try community detection, shortest paths, and triangle counting
4. **Optimize Performance**: Profile your graph operations and optimize based on your data

## Contributing

For questions or improvements to this guide, please open an issue in the repository.

---

*This README complements the main README.md by providing a Databricks-specific, scalable approach to graph processing for RAG applications.*
