import  { useState } from "react";
import { Button, Card, Table } from "react-bootstrap";
import ComparisonGraph from "../comparison/ComparisonGraph";
import { PlusCircle, DashCircle } from 'react-bootstrap-icons';



const ComparisonHistory = ({ research }) => {
    const [activeIndexes, setActiveIndexes] = useState(research?.comparisons?.map((_, index) => index) || []);

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


    const toggleComparison = (index) => {
        setActiveIndexes(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    };

    return (
        <>
            {research?.comparisons?.length > 0 ?
                <Card className="comparison-history">
                    <Card.Header className="d-flex justify-content-between align-items-center">
                        <h3 className="m-0 fw-bold">Comparisons History</h3>
                        <p>Length: {research.comparisons.length}</p>
                    </Card.Header>
                    <Card.Body>
                        {research?.comparisons.map((comparison, index) => (
                            <div className="comparison-data" key={index}>
                                <div className="d-flex align-items-center justify-content-between mb-3 p-2 border-bottom">
                                    <h4 className="mb-0">Comparison #{index + 1}</h4>
                                    <Button
                                        variant="link"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            toggleComparison(index);
                                        }}
                                        style={{ cursor: "pointer" }}
                                    >
                                        {activeIndexes.includes(index) ? (
                                            <DashCircle size={24} className="text-black" />
                                        ) : (
                                            <PlusCircle size={24} className="text-black" />
                                        )}
                                    </Button>
                                </div>
                                {activeIndexes.includes(index) && (
                                    <>
                                        <ComparisonGraph
                                            graphData={research?.analysis}
                                            width={600}
                                            height={500}
                                            isComparisonGraph={false}
                                        />
                                        <ComparisonGraph
                                            graphData={comparison}
                                            width={600}
                                            height={500}
                                            isComparisonGraph={true}
                                        />
                                        <Card className="history-table">
                                            <Card.Header className="card-header-black">
                                                <h5 className="fw-bold">Comparison Statistics</h5>
                                            </Card.Header>
                                            <Card.Body>
                                                <div key={`stats-${index}`} className="mb-4">
                                                    {(() => {
                                                        // const compData = research.comparisons[index];
                                                        const stats = calculateComparisonStats(
                                                            research?.analysis,
                                                            comparison
                                                        );

                                                        if (!stats)
                                                            return (
                                                                <p>Could not calculate comparison statistics.</p>
                                                            );

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
                                            </Card.Body>
                                        </Card>
                                    </>
                                )}
                            </div>
                        ))}
                    </Card.Body>
                </Card>
                :
                <div className="text-center mt-5">
                    <h4>No comparisons available.</h4>
                </div>
            }
        </>
    );
};

export default ComparisonHistory;
