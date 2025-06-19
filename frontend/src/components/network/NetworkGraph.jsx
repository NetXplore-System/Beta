import React, { useRef, useEffect, useMemo } from "react";
import { ForceGraph2D } from "react-force-graph";
import PropTypes from "prop-types";
import * as d3 from "d3-force";
const NetworkGraph = ({
  networkData,
  filteredNodes = [],
  filteredLinks = [],
  customizedNetworkData,
  selectedMetric,
  highlightCentralNodes = false,
  showMetrics = true,
  visualizationSettings = {},
  handleNodeClick,
  networkWasRestored = false,
  forceGraphRef,
  isDirectedGraph = false,
}) => {
  const graphParams = useMemo(() => {
  const nodes = customizedNetworkData
    ? customizedNetworkData.nodes
    : filteredNodes;

  const nodeCount = nodes.length;

  const baseNodeSize = Math.max(10, Math.min(24, 100 / Math.sqrt(nodeCount)));

  const linkDistance = Math.max(
    120,
    Math.min(300, 100 + nodeCount * 2 + baseNodeSize * 3)
  );

  const chargeStrength = Math.max(
    -1000,
    Math.min(-100, -200 - nodeCount * 3 - baseNodeSize * 10)
  );

  const fontSize = Math.max(10, Math.min(16, 80 / Math.sqrt(nodeCount)));

  return {
    baseNodeSize,
    linkDistance,
    chargeStrength,
    fontSize,
    nodeCount,
  };
}, [customizedNetworkData, filteredNodes]);


  useEffect(() => {
    if (forceGraphRef.current && networkData && networkData.nodes) {
      const graph = forceGraphRef.current;

      const setupForces = () => {
        try {
          if (typeof graph.d3Force === "function") {
            const chargeForce = graph.d3Force("charge");
            if (chargeForce?.strength) {
              chargeForce.strength(graphParams.chargeStrength);
            }

            const linkForce = graph.d3Force("link");
            if (linkForce?.distance) {
              linkForce.distance(graphParams.linkDistance);
            }

            graph.d3Force("center", d3.forceCenter(0, 0));
            graph.d3ReheatSimulation?.();
          }
        } catch (error) {
          console.error("Error setting forces:", error);
        }
      };

      const timeout = setTimeout(setupForces, 100);
      return () => clearTimeout(timeout);
    }
  }, [networkData, customizedNetworkData, selectedMetric, graphParams]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (forceGraphRef.current) {
        forceGraphRef.current.zoomToFit(150, 50);
      }
    }, 200);
    return () => clearTimeout(timeout);
  }, [networkData]);

  if (!networkData) {
    return <div>No network data available</div>;
  }

  const getNodeRadius = (node) => {
    const baseSize = graphParams.baseNodeSize;

    if (selectedMetric === "PageRank Centrality") {
      return Math.max(
        baseSize * 0.7,
        Math.min(baseSize * 2, node.pagerank * 500)
      );
    } else if (selectedMetric === "Eigenvector Centrality") {
      return Math.max(
        baseSize * 0.7,
        Math.min(baseSize * 2, node.eigenvector * 80)
      );
    } else if (selectedMetric === "Closeness Centrality") {
      return Math.max(
        baseSize * 0.7,
        Math.min(baseSize * 2, node.closeness * 80)
      );
    } else if (selectedMetric === "Betweenness Centrality") {
      return Math.max(
        baseSize * 0.7,
        Math.min(baseSize * 2, node.betweenness * 100)
      );
    } else if (selectedMetric === "Degree Centrality") {
      return Math.max(baseSize * 0.7, Math.min(baseSize * 2, node.degree * 3));
    }

    return node.size || baseSize;
  };

  return (
    <div
      className="graph-wrapper"
      style={{
        maxWidth: "100%",
        width: "100%",
        height: "700px",
        overflow: "hidden",
        position: "relative",
        margin: "0 auto",
      }}
    >
      <ForceGraph2D
        ref={forceGraphRef}
        key={`${customizedNetworkData ? "customized" : "default"}-${
          graphParams.nodeCount
        }`}
        graphData={{
          nodes: customizedNetworkData
            ? customizedNetworkData.nodes
            : filteredNodes,
          links: (customizedNetworkData
            ? customizedNetworkData.links
            : filteredLinks
          ).map((link) => {
            const nodes = customizedNetworkData
              ? customizedNetworkData.nodes
              : filteredNodes;
            const sourceNode = nodes.find((node) =>
              typeof link.source === "object"
                ? node.id === link.source.id
                : node.id === link.source
            );
            const targetNode = nodes.find((node) =>
              typeof link.target === "object"
                ? node.id === link.target.id
                : node.id === link.target
            );
            return {
              source: sourceNode || link.source,
              target: targetNode || link.target,
              weight: link.weight || 1,
            };
          }),
        }}
        directed={isDirectedGraph}
        width={showMetrics ? 1200 : 1500}
        height={700}
        fitView
        fitViewPadding={100}
        nodeAutoColorBy={null}
        linkWidth={(link) =>
          Math.max(1, Math.min(5, Math.sqrt(link.weight || 1)))
        }
        linkColor={() => "rgba(128, 128, 128, 0.6)"}
        enableNodeDrag={true}
        cooldownTicks={Math.min(200, graphParams.nodeCount * 2)}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.8}
        d3ReheatOnLayout={false}
        onNodeClick={handleNodeClick}
        onEngineStop={() => {
          setTimeout(() => {
            forceGraphRef.current?.zoomToFit(150, 50);
          }, 100);
        }}
        onNodeDragEnd={(node) => {
          node.fx = node.x;
          node.fy = node.y;
        }}
        linkCanvasObject={(link, ctx, globalScale) => {
          if (!link.source || !link.target) return;
          const { x: sourceX, y: sourceY } = link.source;
          const { x: targetX, y: targetY } = link.target;
          const dx = targetX - sourceX;
          const dy = targetY - sourceY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist === 0) return;

          const lineWidth = Math.max(
            0.5,
            Math.min(3, Math.sqrt(link.weight || 1))
          );

          ctx.beginPath();
          ctx.moveTo(sourceX, sourceY);
          ctx.lineTo(targetX, targetY);
          ctx.strokeStyle = "rgba(128, 128, 128, 0.6)";
          ctx.lineWidth = lineWidth;
          ctx.stroke();

          const fontSize = Math.max(
            8,
            (graphParams.fontSize * 0.7) / globalScale
          );
          const label = link.weight || "1";

          if (isDirectedGraph) {
            const arrowLength = Math.max(4, 6 / globalScale);
            const arrowAngle = Math.PI / 8;
            const targetRadius = getNodeRadius(link.target);
            const arrowTipX = targetX - (dx * targetRadius) / dist;
            const arrowTipY = targetY - (dy * targetRadius) / dist;
            const angle = Math.atan2(dy, dx);
            const arrowX1 =
              arrowTipX - arrowLength * Math.cos(angle - arrowAngle);
            const arrowY1 =
              arrowTipY - arrowLength * Math.sin(angle - arrowAngle);
            const arrowX2 =
              arrowTipX - arrowLength * Math.cos(angle + arrowAngle);
            const arrowY2 =
              arrowTipY - arrowLength * Math.sin(angle + arrowAngle);

            ctx.beginPath();
            ctx.moveTo(arrowTipX, arrowTipY);
            ctx.lineTo(arrowX1, arrowY1);
            ctx.lineTo(arrowX2, arrowY2);
            ctx.closePath();
            ctx.fillStyle = "rgba(128, 128, 128, 0.8)";
            ctx.fill();

            const labelX =
              arrowTipX - (dx / dist) * (arrowLength + 4) + (-dy / dist) * 2;
            const labelY =
              arrowTipY - (dy / dist) * (arrowLength + 4) + (dx / dist) * 2;

            ctx.font = `bold ${fontSize}px Sans-Serif`;
            ctx.fillStyle = "#333";
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.strokeText(label, labelX, labelY);
            ctx.fillText(label, labelX, labelY);
          } else {
            const labelX = (sourceX + targetX) / 2;
            const labelY = (sourceY + targetY) / 2;

            ctx.font = `bold ${fontSize}px Sans-Serif`;
            ctx.fillStyle = "#333";
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.strokeText(label, labelX, labelY);
            ctx.fillText(label, labelX, labelY);
          }
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const fontSize = graphParams.fontSize / globalScale;
          const radius = getNodeRadius(node);

          ctx.save();
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);

          const isHighlighted = node.highlighted && highlightCentralNodes;

          const nodeColor = isHighlighted
            ? "#ff9900"
            : node.color ||
              (selectedMetric === "PageRank Centrality"
                ? "#ff8c00"
                : selectedMetric === "Eigenvector Centrality"
                ? "#9932cc"
                : selectedMetric === "Closeness Centrality"
                ? "#32cd32"
                : selectedMetric === "Betweenness Centrality"
                ? "#dc143c"
                : selectedMetric === "Degree Centrality"
                ? "#4169e1"
                : "#050d2d");

          ctx.fillStyle = nodeColor;
          ctx.fill();

          if (node.isHighlightedCommunity) {
            ctx.shadowColor = "#F5BD20";
            ctx.shadowBlur = 10;
            ctx.strokeStyle = "#F5BD20";
            ctx.lineWidth = 2;
            ctx.stroke();
          } else if (isHighlighted) {
            ctx.strokeStyle = "#ffff00";
            ctx.lineWidth = 3;
            ctx.stroke();
          }

          ctx.font = `bold ${fontSize}px Sans-Serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = "white";
          ctx.strokeStyle = "rgba(0,0,0,0.7)";
          ctx.lineWidth = 0.5;
          ctx.strokeText(node.id, node.x, node.y);
          ctx.fillText(node.id, node.x, node.y);

          const metricFontSize = fontSize * 1.1;
          ctx.font = `${metricFontSize}px Sans-Serif`;
          const yOffset = radius + metricFontSize + 2;

          if (selectedMetric === "Degree Centrality") {
            ctx.fillStyle = "#4169e1";
            ctx.strokeStyle = "white";
            ctx.strokeText(`Deg: ${node.degree}`, node.x, node.y + yOffset);
            ctx.fillText(`Deg: ${node.degree}`, node.x, node.y + yOffset);
          } else if (selectedMetric === "Betweenness Centrality") {
            ctx.fillStyle = "#dc143c";
            ctx.strokeStyle = "white";
            ctx.strokeText(
              `Btw: ${node.betweenness?.toFixed(2) || 0}`,
              node.x,
              node.y + yOffset
            );
            ctx.fillText(
              `Btw: ${node.betweenness?.toFixed(2) || 0}`,
              node.x,
              node.y + yOffset
            );
          } else if (selectedMetric === "Closeness Centrality") {
            ctx.fillStyle = "#32cd32";
            ctx.strokeStyle = "white";
            ctx.strokeText(
              `Cls: ${node.closeness?.toFixed(2) || 0}`,
              node.x,
              node.y + yOffset
            );
            ctx.fillText(
              `Cls: ${node.closeness?.toFixed(2) || 0}`,
              node.x,
              node.y + yOffset
            );
          } else if (selectedMetric === "Eigenvector Centrality") {
            ctx.fillStyle = "#9932cc";
            ctx.strokeStyle = "white";
            ctx.strokeText(
              `Eig: ${node.eigenvector?.toFixed(4) || 0}`,
              node.x,
              node.y + yOffset
            );
            ctx.fillText(
              `Eig: ${node.eigenvector?.toFixed(4) || 0}`,
              node.x,
              node.y + yOffset
            );
          } else if (selectedMetric === "PageRank Centrality") {
            ctx.fillStyle = "#ff8c00";
            ctx.strokeStyle = "white";
            ctx.strokeText(
              `PR: ${node.pagerank?.toFixed(4) || 0}`,
              node.x,
              node.y + yOffset
            );
            ctx.fillText(
              `PR: ${node.pagerank?.toFixed(4) || 0}`,
              node.x,
              node.y + yOffset
            );
          }

          ctx.restore();
        }}
        onNodeHover={(node) => {
          document.body.style.cursor =
            node && networkWasRestored ? "pointer" : "default";
        }}
      />
    </div>
  );
};

NetworkGraph.propTypes = {
  networkData: PropTypes.object,
  filteredNodes: PropTypes.array,
  filteredLinks: PropTypes.array,
  customizedNetworkData: PropTypes.object,
  selectedMetric: PropTypes.string,
  highlightCentralNodes: PropTypes.bool,
  showMetrics: PropTypes.bool,
  visualizationSettings: PropTypes.object,
  handleNodeClick: PropTypes.func,
  networkWasRestored: PropTypes.bool,
  forceGraphRef: PropTypes.object,
  isDirectedGraph: PropTypes.bool,
};

export default NetworkGraph;
