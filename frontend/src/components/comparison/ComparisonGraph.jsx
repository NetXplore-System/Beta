import { ForceGraph2D } from "react-force-graph";
import { Card, Button } from "react-bootstrap";
import { GraphContainer } from "../../pages/Form.style.js";
import React, { useState, useRef } from "react";


const ComparisonGraph = ({
  graphData,
  title,
  width = 600,
  height = 500,
  isComparisonGraph = false,
  selectedMetric = null,
  comparisonMetrics = [],
  buttonElement = null,
}) => {
  const [nodesFixed, setNodesFixed] = useState(false);
  const forceGraphRef = useRef(null);

  if (!graphData || !graphData.nodes || !graphData.links) {
    return <div>No graph data available</div>;
  }
  const processedData = {
    nodes: [...graphData.nodes],
    links: graphData.links.map((link) => {
      const sourceId =
        typeof link.source === "object" ? link.source.id : link.source;
      const targetId =
        typeof link.target === "object" ? link.target.id : link.target;

      const sourceNode = graphData.nodes.find((n) => n.id === sourceId);
      const targetNode = graphData.nodes.find((n) => n.id === targetId);

      return {
        source: sourceNode || sourceId,
        target: targetNode || targetId,
        weight: link.weight || 1,
      };
    }),
  };
  const unfixAllNodes = () => {
    if (processedData && processedData.nodes) {
      const hasFixedNodes = processedData.nodes.some(
        (node) => node.fx !== null || node.fy !== null
      );

      if (hasFixedNodes) {
        processedData.nodes.forEach(node => {
          node.fx = null;
          node.fy = null;
        });

        if (forceGraphRef.current) {
          forceGraphRef.current.d3ReheatSimulation();
        }

        setNodesFixed(false);
      }
    }
  };

  const linkColor = isComparisonGraph
    ? "rgba(128, 0, 128, 0.6)"
    : "rgba(128, 128, 128, 0.6)";

  return (
    <Card>
          <Card.Header as="h5" className="d-flex justify-content-between align-items-center">
        <span>{title}</span>
        {nodesFixed && (
          <Button 
            variant="outline-secondary" 
            size="sm" 
            onClick={unfixAllNodes}
          >
            Release Nodes
          </Button>
        )}
      </Card.Header>
      <Card.Body className="text-center">
        <GraphContainer>
          {buttonElement}
          <ForceGraph2D
            graphData={processedData}
            width={width}
            height={height}
            fitView
            fitViewPadding={20}
            nodeAutoColorBy="id"
            onNodeDragEnd={(node) => {
              node.fx = node.x;
              node.fy = node.y;
              setNodesFixed(true);
            }}
            linkWidth={(link) => Math.sqrt(link.weight || 1)}
            linkColor={() => linkColor}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const fontSize = 12 / globalScale;

              let nodeSize = 20;
              let nodeColor = isComparisonGraph ? "purple" : "blue";

              if (comparisonMetrics.length > 0) {
                if (comparisonMetrics.includes("Degree Centrality")) {
                  nodeSize = Math.max(10, (node.degree || 0) * 80);
                  nodeColor = "#231d81";
                }
                if (comparisonMetrics.includes("Betweenness Centrality")) {
                  nodeSize = Math.max(10, (node.betweenness || 0) * 80);
                  nodeColor = "red";
                }
                if (comparisonMetrics.includes("Closeness Centrality")) {
                  nodeSize = Math.max(10, (node.closeness || 0) * 50);
                  nodeColor = "green";
                }
                if (comparisonMetrics.includes("Eigenvector Centrality")) {
                  nodeSize = Math.max(10, (node.eigenvector || 0) * 60);
                  nodeColor = "purple";
                }
                if (comparisonMetrics.includes("PageRank Centrality")) {
                  nodeSize = Math.max(10, (node.pagerank || 0) * 500);
                  nodeColor = "orange";
                }
              }
              else if (selectedMetric) {
                if (selectedMetric === "Degree Centrality") {
                  nodeSize = Math.max(10, (node.degree || 0) * 80);
                  nodeColor = "#231d81";
                } else if (selectedMetric === "Betweenness Centrality") {
                  nodeSize = Math.max(10, (node.betweenness || 0) * 80);
                  nodeColor = "red";
                } else if (selectedMetric === "Closeness Centrality") {
                  nodeSize = Math.max(10, (node.closeness || 0) * 50);
                  nodeColor = "green";
                } else if (selectedMetric === "Eigenvector Centrality") {
                  nodeSize = Math.max(10, (node.eigenvector || 0) * 60);
                  nodeColor = "purple";
                } else if (selectedMetric === "PageRank Centrality") {
                  nodeSize = Math.max(10, (node.pagerank || 0) * 500);
                  nodeColor = "orange";
                }
              }

              ctx.save();
              ctx.beginPath();
              ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI, false);
              ctx.fillStyle = nodeColor;
              ctx.fill();

              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillStyle = "white";
              ctx.fillText(node.id, node.x, node.y);

              if (comparisonMetrics.length > 0) {
                if (comparisonMetrics.includes("Degree Centrality")) {
                  ctx.fillStyle = "#231d81";
                  ctx.fillText(
                    `Deg: ${node.degree || 0}`,
                    node.x,
                    node.y + nodeSize + 5
                  );
                } else if (
                  comparisonMetrics.includes("Betweenness Centrality")
                ) {
                  ctx.fillStyle = "DarkRed";
                  ctx.fillText(
                    `Btw: ${(node.betweenness || 0).toFixed(2)}`,
                    node.x,
                    node.y + nodeSize + 5
                  );
                } else if (comparisonMetrics.includes("Closeness Centrality")) {
                  ctx.fillStyle = "green";
                  ctx.fillText(
                    `Cls: ${(node.closeness || 0).toFixed(2)}`,
                    node.x,
                    node.y + nodeSize + 5
                  );
                } else if (
                  comparisonMetrics.includes("Eigenvector Centrality")
                ) {
                  ctx.fillStyle = "purple";
                  ctx.fillText(
                    `Eig: ${(node.eigenvector || 0).toFixed(4)}`,
                    node.x,
                    node.y + nodeSize + 5
                  );
                } else if (comparisonMetrics.includes("PageRank Centrality")) {
                  ctx.fillStyle = "orange";
                  ctx.fillText(
                    `PR: ${(node.pagerank || 0).toFixed(4)}`,
                    node.x,
                    node.y + nodeSize + 5
                  );
                }
              }
              // For single metric mode
              else if (selectedMetric === "Degree Centrality") {
                ctx.fillStyle = "#231d81";
                ctx.fillText(
                  `Deg: ${node.degree || 0}`,
                  node.x,
                  node.y + nodeSize + 5
                );
              } else if (selectedMetric === "Betweenness Centrality") {
                ctx.fillStyle = "DarkRed";
                ctx.fillText(
                  `Btw: ${(node.betweenness || 0).toFixed(2)}`,
                  node.x,
                  node.y + nodeSize + 5
                );
              } else if (selectedMetric === "Closeness Centrality") {
                ctx.fillStyle = "green";
                ctx.fillText(
                  `Cls: ${(node.closeness || 0).toFixed(2)}`,
                  node.x,
                  node.y + nodeSize + 5
                );
              } else if (selectedMetric === "Eigenvector Centrality") {
                ctx.fillStyle = "purple";
                ctx.fillText(
                  `Eig: ${(node.eigenvector || 0).toFixed(4)}`,
                  node.x,
                  node.y + nodeSize + 5
                );
              } else if (selectedMetric === "PageRank Centrality") {
                ctx.fillStyle = "orange";
                ctx.fillText(
                  `PR: ${(node.pagerank || 0).toFixed(4)}`,
                  node.x,
                  node.y + nodeSize + 5
                );
              }
              ctx.restore();
            }}
            linkCanvasObject={(link, ctx, globalScale) => {
              if (!link.source || !link.target) return;

              // Draw the link
              ctx.beginPath();
              ctx.moveTo(link.source.x, link.source.y);
              ctx.lineTo(link.target.x, link.target.y);
              ctx.strokeStyle = linkColor;
              ctx.lineWidth = Math.sqrt(link.weight || 1);
              ctx.stroke();

              // Draw the link weight
              const midX = (link.source.x + link.target.x) / 2;
              const midY = (link.source.y + link.target.y) / 2;
              const fontSize = 10 / globalScale;
              ctx.save();
              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.fillStyle = "black";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(link.weight || "1", midX, midY);
              ctx.restore();
            }}
          />
        </GraphContainer>
      </Card.Body>
    </Card>
  );
};

export default ComparisonGraph;
