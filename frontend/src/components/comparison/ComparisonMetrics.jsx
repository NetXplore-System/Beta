import React, { useState } from "react";
import { Card, Row, Col, Table, Form } from "react-bootstrap";
import ComparisonGraph from "./ComparisonGraph";
import { useDispatch } from "react-redux";
import { clearTableData, setTableData } from "../../redux/table/tableSlice";
import { addToComparison } from "../../redux/images/imagesSlice";

const ComparisonMetrics = ({
  originalNetworkData,
  comparisonNetworkData,
  comparisonData,
  activeComparisonIndices,
  onToggleComparison,
  filteredOriginalData,
  filteredComparisonData,
  onApplyComparisonFilters,
  onResetComparisonFilters,
}) => {

  const dispatch = useDispatch();
  const [comparisonFilter, setComparisonFilter] = useState("");
  const [minComparisonWeight, setMinComparisonWeight] = useState(1);
  const [comparisonMetrics, setComparisonMetrics] = useState([]);
  const [highlightCommonNodes, setHighlightCommonNodes] = useState(false);

  const graphMetrics = [
    "Degree Centrality",
    "Betweenness Centrality",
    "Closeness Centrality",
    "Eigenvector Centrality",
    "PageRank Centrality",
  ];

  const calculateComparisonStats = (originalData, comparisonData) => {
    if (!originalData || !comparisonData) {
      return null;
    }

    const originalNodeCount = originalData.nodes.length;
    const comparisonNodeCount = comparisonData.nodes.length;
    const originalLinkCount = originalData.links.length;
    const comparisonLinkCount = comparisonData.links.length;

    const nodeDifference = comparisonNodeCount - originalNodeCount;
    const linkDifference = comparisonLinkCount - originalLinkCount;

    const nodeChangePercent = originalNodeCount
      ? (
        ((comparisonNodeCount - originalNodeCount) / originalNodeCount) *
        100
      ).toFixed(2)
      : 0;
    const linkChangePercent = originalLinkCount
      ? (
        ((comparisonLinkCount - originalLinkCount) / originalLinkCount) *
        100
      ).toFixed(2)
      : 0;

    const originalNodeIds = new Set(originalData.nodes.map((node) => node.id));
    const comparisonNodeIds = new Set(
      comparisonData.nodes.map((node) => node.id)
    );

    const commonNodes = [...originalNodeIds].filter((id) =>
      comparisonNodeIds.has(id)
    );
    const commonNodesCount = commonNodes.length;

    return {
      originalNodeCount,
      comparisonNodeCount,
      originalLinkCount,
      comparisonLinkCount,
      nodeDifference,
      linkDifference,
      nodeChangePercent,
      linkChangePercent,
      commonNodesCount,
    };
  };

  const toggleComparisonMetric = (metric) => {
    setComparisonMetrics((prev) =>
      prev.includes(metric)
        ? prev.filter((m) => m !== metric)
        : [...prev, metric]
    );
    setTimeout(() => {
      handleApplyFilters();
    }, 0);
  };

  const handleApplyFilters = () => {
    if (onApplyComparisonFilters) {
      onApplyComparisonFilters({
        comparisonFilter,
        minComparisonWeight,
        comparisonMetrics,
        highlightCommonNodes,
      });
    }
  };

  const handleResetFilters = () => {
    setComparisonFilter("");
    setMinComparisonWeight(1);
    setComparisonMetrics([]);
    setHighlightCommonNodes(false);
    if (onResetComparisonFilters) {
      onResetComparisonFilters();
    }
  };

  // Only render active comparison items
  const activeComparisons = comparisonNetworkData.filter((data, idx) =>
    activeComparisonIndices.includes(idx)
  );

  dispatch(clearTableData());

  const handleScreenshot = (e, source, index, i) => {
    e.stopPropagation();
    const canvas = document.querySelectorAll("canvas");
    source ?
      dispatch(addToComparison({
        data: canvas[1].toDataURL('image/png'),
        type: "source",
        width: index > 1 ? "big" : "small",
      }))
      :
      dispatch(addToComparison({
        data: canvas[i + 2].toDataURL('image/png'),
        type: "comparison",
        index
      }))

  };


  return (
    <>
      {comparisonNetworkData.map((data, index) => {
        if (data && data.nodes && data.links) {
          return (
            <Col md={12} key={index}>
              <Card className="mt-3 mb-3">
                <Card.Header as="h5" className="d-flex justify-content-between">
                  <span>
                    Comparison #{index + 1}: {comparisonData[index]?.name || ""}
                  </span>
                  <Form.Check
                    type="checkbox"
                    label="Add to comparison view"
                    checked={activeComparisonIndices.includes(index)}
                    onChange={() => onToggleComparison(index)}
                    className="mt-1"
                  />
                </Card.Header>
              </Card>
            </Col>
          );
        }
        return null;
      })}

      {activeComparisonIndices.length > 0 && originalNetworkData && (
        <>
          <Row className="mt-4">
            <Card>
              <Card.Header>
                <h5 className="fw-bold">Network Comparison View</h5>
              </Card.Header>
              <Card.Body>
                <div className="comparison-toolbar mb-3">
                  <div className="toolbar-section">
                    <span className="toolbar-label">Filter:</span>
                    <input
                      type="text"
                      className="toolbar-input"
                      placeholder="Filter nodes..."
                      value={comparisonFilter}
                      onChange={(e) => setComparisonFilter(e.target.value)}
                    />
                  </div>

                  <div className="toolbar-section">
                    <span className="toolbar-label">Min Weight:</span>
                    <input
                      type="number"
                      className="toolbar-input"
                      min="1"
                      value={minComparisonWeight}
                      onChange={(e) =>
                        setMinComparisonWeight(parseInt(e.target.value) || 1)
                      }
                    />
                  </div>

                  <div className="toolbar-section metrics-toggles">
                    <span className="toolbar-label">Metrics:</span>
                    <div className="toolbar-buttons">
                      {graphMetrics.map((metric) => (
                        <button
                          key={metric}
                          className={`toolbar-button ${comparisonMetrics.includes(metric) ? "active" : ""
                            }`}
                          onClick={() => toggleComparisonMetric(metric)}
                          title={metric}
                        >
                          {metric.split(" ")[0]}
                        </button>
                      ))}
                      <button
                        className={`toolbar-button ${highlightCommonNodes ? "active" : ""
                          }`}
                        onClick={() =>
                          setHighlightCommonNodes(!highlightCommonNodes)
                        }
                        title="Highlight Common Nodes"
                      >
                        Common
                      </button>
                    </div>
                  </div>

                  <div className="toolbar-section">
                    <button
                      className="toolbar-action-btn"
                      onClick={handleApplyFilters}
                    >
                      Apply
                    </button>
                    <button
                      className="toolbar-action-btn outline"
                      onClick={handleResetFilters}
                    >
                      Reset
                    </button>
                  </div>
                </div>

                <Row>
                  <Col
                    md={activeComparisonIndices.length > 1 ? 12 : 6}
                    className="mb-4"
                  >
                    <ComparisonGraph
                      graphData={filteredOriginalData || originalNetworkData}
                      title="Original Network"
                      width={activeComparisonIndices.length > 1 ? 1000 : 600}
                      height={500}
                      comparisonMetrics={comparisonMetrics}
                      buttonElement={
                        <button className="graph-button" onClick={(e) => handleScreenshot(e, true, activeComparisonIndices.length)}>Take Screenshot</button>
                      }
                    />
                  </Col>

                  {activeComparisonIndices.map((index, i) => (
                    <Col
                      md={activeComparisonIndices.length > 1 ? 6 : 6}
                      key={`comparison-${index}`}
                      className="mb-4"
                    >
                      <ComparisonGraph
                        graphData={
                          filteredComparisonData &&
                            filteredComparisonData[index]
                            ? filteredComparisonData[index]
                            : comparisonNetworkData[index]
                        }
                        title={`Comparison Network #${index + 1}`}
                        width={activeComparisonIndices.length > 2 ? 600 : 600}
                        height={500}
                        isComparisonGraph={true}
                        comparisonMetrics={comparisonMetrics}
                        buttonElement={
                          <button className="graph-button" onClick={(e) => handleScreenshot(e, false, index, i)}>Take Screenshot</button>
                        }
                      />
                    </Col>
                  ))}
                </Row>
              </Card.Body>
            </Card>
          </Row>

          <Row className="mt-4 mb-4">
            <Card>
              <Card.Header>
                <h5 className="fw-bold">Comparison Statistics</h5>
              </Card.Header>
              <Card.Body>
                {activeComparisonIndices.map((index) => {
                  console.log(comparisonData[index]?.name);

                  return (
                    <div key={`stats-${index}`} className="mb-4">
                      <h6>
                        Statistics for Comparison #{index + 1}:{" "}
                        {comparisonData[index]?.name || ""}
                      </h6>
                      {(() => {
                        const compData = comparisonNetworkData[index];
                        const stats = calculateComparisonStats(
                          originalNetworkData,
                          compData
                        );

                        if (!stats)
                          return (
                            <p>Could not calculate comparison statistics.</p>
                          );

                        dispatch(setTableData({ ...stats, index, fileName: comparisonData[index]?.name || "unknon name" }));


                        return (
                          <Table responsive striped bordered hover >
                            <thead>
                              <tr>
                                <th>Metric</th>
                                <th>Original Network</th>
                                <th>Comparison Network</th>
                                <th>Difference</th>
                                <th>Change %</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td>Node Count</td>
                                <td>{stats.originalNodeCount}</td>
                                <td>{stats.comparisonNodeCount}</td>
                                <td>
                                  {stats.nodeDifference > 0
                                    ? `+${stats.nodeDifference}`
                                    : stats.nodeDifference}
                                </td>
                                <td>{stats.nodeChangePercent}%</td>
                              </tr>
                              <tr>
                                <td>Edge Count</td>
                                <td>{stats.originalLinkCount}</td>
                                <td>{stats.comparisonLinkCount}</td>
                                <td>
                                  {stats.linkDifference > 0
                                    ? `+${stats.linkDifference}`
                                    : stats.linkDifference}
                                </td>
                                <td>{stats.linkChangePercent}%</td>
                              </tr>
                              <tr>
                                <td>Common Nodes</td>
                                <td colSpan="2">{stats.commonNodesCount}</td>
                                <td colSpan="2">
                                  {(
                                    (stats.commonNodesCount /
                                      stats.originalNodeCount) *
                                    100
                                  ).toFixed(2)}
                                  % of original network
                                </td>
                              </tr>
                            </tbody>
                          </Table>
                        );
                      })()}
                    </div>
                  )
                })}
              </Card.Body>
            </Card>
          </Row>
        </>
      )}
    </>
  );
};

export default ComparisonMetrics;
