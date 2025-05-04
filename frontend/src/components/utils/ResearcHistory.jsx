import { FaWhatsapp, FaWikipediaW } from 'react-icons/fa';
import { Row, Col, Button, Card } from 'react-bootstrap';
import { ChevronLeft, ChevronRight, FileBarGraph } from 'react-bootstrap-icons';
import MetricsButton from '../common/MetricsButton';
import { graphMetrics } from '../../pages/Home';
import { toast } from 'react-hot-toast';
import { useState, useEffect, useRef } from 'react';
import MetricsPanel from '../network/MetricsPanel';
import NetworkCustomizationToolbar from '../NetworkCustomizationToolbar';
import NetworkGraph from '../network/NetworkGraph';
import NetworkDataTable from '../NetworkDataTable';




const ResearchHistory = ({ research }) => {
    const [showMetrics, setShowMetrics] = useState(true);
    const [showDataTable, setShowDataTable] = useState(false);
    const [strongConnectionsActive, setStrongConnectionsActive] = useState(false);
    const [highlightCentralNodes, setHighlightCentralNodes] = useState(false);
    const [networkWasRestored, setNetworkWasRestored] = useState(false);
    const [activityFilterEnabled, setActivityFilterEnabled] = useState(false);
    const [showOnlyIntraCommunityLinks, setShowOnlyIntraCommunityLinks] = useState(false);
    const [selectedMetric, setSelectedMetric] = useState(null);
    const [showDensity, setShowDensity] = useState(false);
    const [densityValue, setDensityValue] = useState(0);
    const [showDiameter, setShowDiameter] = useState(false);
    const [diameterValue, setDiameterValue] = useState(0);
    const [networkStats, setNetworkStats] = useState({});
    const [originalNetworkData, setOriginalNetworkData] = useState(research.analysis ? research.analysis : null);
    const [networkData, setNetworkData] = useState(research.analysis ? research.analysis : null);
    const [communities, setCommunities] = useState([]);
    const [communityMap, setCommunityMap] = useState({});
    const [customizedNetworkData, setCustomizedNetworkData] = useState(null);
    const [visualizationSettings, setVisualizationSettings] = useState({
        colorBy: "default",
        sizeBy: "default",
        highlightUsers: [],
        highlightCommunities: [],
        communityNames: {},
        communityColors: {},
        customColors: {
            defaultNodeColor: "#050d2d",
            highlightNodeColor: "#00c6c2",
            communityColors: [
                "#313659",
                "#5f6289",
                "#324b4a",
                "#158582",
                "#9092bc",
                "#c4c6f1",
            ],
            edgeColor: "rgba(128, 128, 128, 0.6)",
        },
        nodeSizes: {
            min: 15,
            max: 40,
        },
        colorScheme: "default",
        showImportantNodes: false,
        importantNodesThreshold: 0.5,
    });
    const [comparisonData, setComparisonData] = useState(research?.comparisons || []);
    const [filtersData, setFiltersData] = useState({
        nodeFilter: '',
        minWeight: 1,
        metrics: []
    });
    const forceGraphRef = useRef(null);


    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-GB', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const filters = research.filters ? Object.entries(research.filters)
        .filter(([_, value]) => value !== null && value !== undefined && _ !== 'filter_id' && _ !== 'research_id')
        .map(([key, value]) => ({
            label: key
                .split('_')
                .map(word => word[0].toUpperCase() + word.slice(1))
                .join(' '),
            value
        })) : [];


    const handleToggleMetric = (metric) => {
        setSelectedMetric(selectedMetric === metric ? null : metric);
    };

    const handleDensityMetric = () => {
        const density = calculateDensity(networkData.nodes, networkData.links);
        setDensityValue(density.toFixed(4));
        setShowDensity(!showDensity);
    };

    const handleDiameterMetric = () => {
        setShowDiameter(!showDiameter);
        if (!showDiameter && networkData) {
            const diameter = calculateDiameter(networkData.nodes, networkData.links);
            setDiameterValue(diameter);
        }
    };

    const handleStrongConnections = () => {
        if (!networkData) return;

        if (strongConnectionsActive) {
            setNetworkData(originalNetworkData);
            setStrongConnectionsActive(false);

            setNetworkWasRestored(false);
        } else {
            const threshold = 0.2;
            const filteredNodes = networkData.nodes.filter(
                (node) => node.betweenness >= threshold
            );
            const filteredLinks = networkData.links.filter(
                (link) =>
                    filteredNodes.some((node) => node.id === link.source) &&
                    filteredNodes.some((node) => node.id === link.target)
            );

            setNetworkData({ nodes: filteredNodes, links: filteredLinks });
            setStrongConnectionsActive(true);
        }
    };

    const handleHighlightCentralNodes = () => {
        if (!networkData) return;

        setHighlightCentralNodes(!highlightCentralNodes);

        setNetworkWasRestored(false);

        if (!highlightCentralNodes) {
            let centralityMeasure = "degree";
            if (selectedMetric === "Betweenness Centrality")
                centralityMeasure = "betweenness";
            else if (selectedMetric === "Closeness Centrality")
                centralityMeasure = "closeness";
            else if (selectedMetric === "Eigenvector Centrality")
                centralityMeasure = "eigenvector";
            else if (selectedMetric === "PageRank Centrality")
                centralityMeasure = "pagerank";

            setNetworkWasRestored(false);

            const updatedNodes = networkData.nodes.map((node) => {
                const centralityValue = node[centralityMeasure] || 0;
                const sortedNodes = [...networkData.nodes].sort(
                    (a, b) => (b[centralityMeasure] || 0) - (a[centralityMeasure] || 0)
                );

                const thresholdIndex = Math.floor(sortedNodes.length * 0.2);
                const threshold =
                    thresholdIndex >= 0 && thresholdIndex < sortedNodes.length
                        ? sortedNodes[thresholdIndex][centralityMeasure] || 0
                        : 0;

                return {
                    ...node,
                    highlighted: centralityValue >= threshold,
                };
            });

            setNetworkData({
                ...networkData,
                nodes: updatedNodes,
            });
        } else {
            const resetNodes = networkData.nodes.map((node) => ({
                ...node,
                highlighted: false,
            }));

            setNetworkData({
                ...networkData,
                nodes: resetNodes,
            });
        }
    };

    const handleActivityFilter = () => {
        if (!networkData) return;

        const newState = !activityFilterEnabled;
        setActivityFilterEnabled(newState);

        if (newState) {
            if (!originalNetworkData) {
                setOriginalNetworkData(networkData);
                setNetworkWasRestored(false);
            }

            const connectionCounts = {};
            networkData.nodes.forEach((node) => {
                connectionCounts[node.id] = 0;
            });

            networkData.links.forEach((link) => {
                const sourceId =
                    typeof link.source === "object" ? link.source.id : link.source;
                const targetId =
                    typeof link.target === "object" ? link.target.id : link.target;

                connectionCounts[sourceId] = (connectionCounts[sourceId] || 0) + 1;
                connectionCounts[targetId] = (connectionCounts[targetId] || 0) + 1;
            });

            const activeNodes = networkData.nodes.filter(
                (node) => connectionCounts[node.id] >= activityThreshold
            );

            const activeLinks = networkData.links.filter((link) => {
                const sourceId =
                    typeof link.source === "object" ? link.source.id : link.source;
                const targetId =
                    typeof link.target === "object" ? link.target.id : link.target;

                return (
                    activeNodes.some((node) => node.id === sourceId) &&
                    activeNodes.some((node) => node.id === targetId)
                );
            });

            setNetworkData({
                nodes: activeNodes,
                links: activeLinks,
            });
        } else {
            if (originalNetworkData) {
                setNetworkData(originalNetworkData);
                setNetworkWasRestored(true);
            }
        }
    };


    const handleRestoreNetwork = () => {
        if (originalNetworkData) {
            if (networkWasRestored) {
                setNetworkWasRestored(false);
            } else {
                setNetworkData(originalNetworkData);
                setNetworkWasRestored(true);

                if (activityFilterEnabled) {
                    setActivityFilterEnabled(false);
                }
            }
        }
    };

    const handleToggleCommunitiesFilter = () => {
        if (!networkData || !originalNetworkData) return;

        if (!communityMap || Object.keys(communityMap).length === 0) {
            toast.error("Community data not found. Detecting communities...");
            detectAndApplyCommunityData();
            return;
        }

        const newState = !showOnlyIntraCommunityLinks;
        setShowOnlyIntraCommunityLinks(newState);

        if (newState) {
            const filteredLinks = networkData.links.filter((link) => {
                const sourceId =
                    typeof link.source === "object" ? link.source.id : link.source;
                const targetId =
                    typeof link.target === "object" ? link.target.id : link.target;

                const sourceCommunity = communityMap[sourceId?.toString().trim()];
                const targetCommunity = communityMap[targetId?.toString().trim()];

                if (sourceCommunity === undefined || targetCommunity === undefined) {
                    return false;
                }

                return sourceCommunity === targetCommunity;
            });

            const connectedNodeIds = new Set();
            filteredLinks.forEach((link) => {
                const sourceId =
                    typeof link.source === "object" ? link.source.id : link.source;
                const targetId =
                    typeof link.target === "object" ? link.target.id : link.target;
                connectedNodeIds.add(sourceId);
                connectedNodeIds.add(targetId);
            });

            const communities = [...new Set(Object.values(communityMap))];
            const radius = 500;
            const angleStep = (2 * Math.PI) / communities.length;

            const communityCenters = {};
            communities.forEach((community, index) => {
                const angle = index * angleStep;
                communityCenters[community] = {
                    x: radius * Math.cos(angle),
                    y: radius * Math.sin(angle),
                };
            });

            const communityColors = [
                "#313659",
                "#5f6289",
                "#324b4a",
                "#158582",
                "#9092bc",
                "#c4c6f1",
                "#ff9800",
                "#4caf50",
                "#2196f3",
                "#e91e63",
                "#9c27b0",
                "#795548",
            ];

            const updatedNodes = networkData.nodes
                .filter((node) => connectedNodeIds.has(node.id))
                .map((node) => {
                    const community = communityMap[node.id?.toString().trim()];
                    const center = communityCenters[community];
                    const jitter = 30;

                    return {
                        ...node,
                        community,
                        originalX: node.x,
                        originalY: node.y,
                        x: center.x + (Math.random() * jitter * 2 - jitter),
                        y: center.y + (Math.random() * jitter * 2 - jitter),
                        color: communityColors[Number(community) % communityColors.length],
                    };
                });

            setNetworkData({
                nodes: updatedNodes,
                links: filteredLinks,
            });

            toast.success(
                `Showing only intra-community links and hiding isolated nodes. Removed ${networkData.links.length - filteredLinks.length
                } cross-community links.`
            );
        } else {
            const restoredNodes = originalNetworkData.nodes.map((node) => {
                const currentNode = networkData.nodes.find((n) => n.id === node.id);
                if (
                    currentNode &&
                    currentNode.originalX !== undefined &&
                    currentNode.originalY !== undefined
                ) {
                    return {
                        ...node,
                        x: currentNode.originalX,
                        y: currentNode.originalY,
                    };
                }
                return node;
            });

            setNetworkData({
                nodes: restoredNodes,
                links: originalNetworkData.links,
            });

            toast.success("Showing all links in the network.");
        }

        if (forceGraphRef.current) {
            setTimeout(() => {
                forceGraphRef.current.d3ReheatSimulation();
                forceGraphRef.current.zoomToFit(400);
            }, 100);
        }
    };

    const calculateNetworkStats = () => {
        if (!networkData) return;

        const { nodes, links } = networkData;
        const numNodes = nodes.length;
        const numEdges = links.length;
        const inDegreeMap = {};
        const outDegreeMap = {};
        let reciprocalEdges = 0;

        links.forEach((link) => {
            inDegreeMap[link.target] = (inDegreeMap[link.target] || 0) + 1;
            outDegreeMap[link.source] = (outDegreeMap[link.source] || 0) + 1;

            if (
                links.some((l) => l.source === link.target && l.target === link.source)
            ) {
                reciprocalEdges++;
            }
        });

        const reciprocity =
            numEdges > 0 ? (reciprocalEdges / numEdges).toFixed(2) : 0;

        setNetworkStats({
            numNodes,
            numEdges,
            reciprocity,
            inDegreeMap,
            outDegreeMap,
        });
    };

    const handleNetworkCustomization = (settings) => {
        setVisualizationSettings(settings);

        if (!networkData) return;

        const customizedNodes = JSON.parse(JSON.stringify(networkData.nodes));
        const customizedLinks = JSON.parse(JSON.stringify(networkData.links));

        for (const node of customizedNodes) {
            let nodeSize = settings.nodeSizes.min;
            let nodeColor = settings.customColors.defaultNodeColor;

            const sizeByValue = (metric) => {
                const maxVal = Math.max(...customizedNodes.map((n) => n[metric] || 0));
                const ratio = maxVal > 0 ? (node[metric] || 0) / maxVal : 0;
                return (
                    settings.nodeSizes.min +
                    ratio * (settings.nodeSizes.max - settings.nodeSizes.min)
                );
            };

            if (settings.sizeBy === "messages") nodeSize = sizeByValue("messages");
            else if (settings.sizeBy === "degree") nodeSize = sizeByValue("degree");
            else if (settings.sizeBy === "betweenness")
                nodeSize = sizeByValue("betweenness");
            else if (settings.sizeBy === "pagerank")
                nodeSize = sizeByValue("pagerank");

            if (settings.colorBy === "community" && node.community !== undefined) {
                const communityId = parseInt(node.community, 10);
                nodeColor =
                    settings.communityColors?.[communityId] ??
                    settings.customColors.communityColors[
                    communityId % settings.customColors.communityColors.length
                    ];
            } else if (settings.colorBy === "degree") {
                const maxDegree = Math.max(
                    ...customizedNodes.map((n) => n.degree || 0)
                );
                const ratio = maxDegree > 0 ? (node.degree || 0) / maxDegree : 0;
                nodeColor = interpolateColor(
                    "#ffefca",
                    settings.customColors.defaultNodeColor,
                    ratio
                );
            } else if (settings.colorBy === "betweenness") {
                const maxBtw = Math.max(
                    ...customizedNodes.map((n) => n.betweenness || 0)
                );
                const ratio = maxBtw > 0 ? (node.betweenness || 0) / maxBtw : 0;
                nodeColor = interpolateColor("#ffefca", "#FF5733", ratio);
            } else if (settings.colorBy === "pagerank") {
                const maxPR = Math.max(...customizedNodes.map((n) => n.pagerank || 0));
                const ratio = maxPR > 0 ? (node.pagerank || 0) / maxPR : 0;
                nodeColor = interpolateColor("#ffefca", "#3366CC", ratio);
            } else if (
                settings.colorBy === "custom" &&
                settings.highlightUsers.includes(node.id)
            ) {
                nodeColor = settings.customColors.highlightNodeColor;
            }

            if (
                settings.highlightCommunities?.includes(parseInt(node.community, 10))
            ) {
                node.isHighlightedCommunity = true;
            } else {
                node.isHighlightedCommunity = false;
            }

            if (settings.showImportantNodes) {
                const threshold = settings.importantNodesThreshold || 0.5;
                const maxBetweenness = Math.max(
                    ...customizedNodes.map((n) => n.betweenness || 0)
                );
                const maxPageRank = Math.max(
                    ...customizedNodes.map((n) => n.pagerank || 0)
                );
                const isImportant =
                    node.betweenness / maxBetweenness > threshold ||
                    node.pagerank / maxPageRank > threshold;

                if (isImportant) {
                    nodeColor = settings.customColors.highlightNodeColor;
                    nodeSize = Math.max(nodeSize, settings.nodeSizes.max * 0.8);
                }
            }

            node.size = nodeSize;
            node.color = nodeColor;
        }

        setCustomizedNetworkData({
            nodes: customizedNodes,
            links: customizedLinks,
        });
    };

    const filteredNodes = networkData ? networkData.nodes : [];


    const filteredLinks = networkData
        ? networkData.links.filter(
            (link) =>
                filteredNodes.some((node) => node.id === link.source) &&
                filteredNodes.some((node) => node.id === link.target)
        )
        : [];

    const handleNodeClick = (node) => {
        if (networkWasRestored) {
            setSelectedNode(node);
            setShowRemoveNodeModal(true);
        }
    };

    const detectAndApplyCommunityData = async () => {

        if (!networkData) return;
        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/history/analyze/communities?algorithm=${research.analysis.algorithm || "louvain"}`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    nodes: networkData.nodes,
                    links: networkData.links,
                }),
            })
            const data = await response.json();
            if (data.communities && data.nodes) {
                setCommunities(data.communities);

                const newCommunityMap = {};

                data.nodes.forEach((node) => {
                    if (node.community !== undefined) {
                        newCommunityMap[node.id.toString().trim()] = node.community;
                    }
                });

                setCommunityMap(newCommunityMap);

                if (networkData && networkData.nodes) {
                    const updatedNodes = networkData.nodes.map((node) => {
                        const normalizedId = node.id.toString().trim();
                        const community = newCommunityMap[normalizedId];

                        if (community !== undefined) {
                            return { ...node, community };
                        }

                        return node;
                    });

                    setNetworkData({
                        nodes: updatedNodes,
                        links: networkData.links,
                    });

                    setOriginalNetworkData({
                        nodes: updatedNodes,
                        links: networkData.links,
                    });

                    toast.success(`Detected ${data.communities.length} communities in the network.`);
                }
            } else {
                toast.error("No community data returned from server.");
            }

        } catch (error) {
            console.error("Error during community detection:", error);
            toast.error("An error occurred during community detection.");
        }
    };

    const fetchCommunityData = async () => {
        if (!networkData) {
            toast.error("Data not found");
            return;
        }

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/history/analyze/communities?algorithm=${research.analysis.algorithm || "louvain"}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    nodes: networkData.nodes,
                    links: networkData.links,
                }),
            })
            const data = await response.json();
            if (data.communities && data.nodes) {
                if (data.communities && data.nodes) {
                    setCommunities(data.communities);

                    const newCommunityMap = {};

                    data.nodes.forEach((node) => {
                        if (node.community !== undefined) {
                            newCommunityMap[node.id.toString().trim()] = node.community;
                        }
                    });

                    setCommunityMap(newCommunityMap);

                    if (networkData && networkData.nodes) {
                        const updatedNodes = networkData.nodes.map((node) => {
                            const normalizedId = node.id.toString().trim();
                            const community = newCommunityMap[normalizedId];

                            if (community !== undefined) {
                                return { ...node, community };
                            }

                            return node;
                        });

                        setNetworkData({
                            nodes: updatedNodes,
                            links: networkData.links,
                        });

                        setOriginalNetworkData({
                            nodes: updatedNodes,
                            links: networkData.links,
                        });

                        toast.success(`Detected ${data.communities.length} communities in the network.`);
                    }
                } else {
                    toast.error("No community data returned from server.");
                }
            }
        } catch (error) {
            console.error("Error during community detection:", error);
            toast.error("An error occurred during community detection.");
        }
    };

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


    const handleNodeFilterChange = (e) => {
        setFiltersData(prev => ({ ...prev, nodeFilter: e.target.value }));
    };

    const handleMinWeightChange = (e) => {
        setFiltersData(prev => ({ ...prev, minWeight: parseInt(e.target.value, 10) }));
    };

    const handleMetricChange = (metric) => {
        setFiltersData(prev => {
            const metrics = prev.metrics.includes(metric)
                ? prev.metrics.filter(m => m !== metric)
                : [...prev.metrics, metric];
            return { ...prev, metrics };
        });
    };

    const handleApply = async (index) => {
        const filters = new URLSearchParams();
        filters.set('node_filter', filtersData.nodeFilter);
        filters.set('min_weight', filtersData.minWeight);
        filters.set('metrics', filtersData.metrics);
        filters.set('research_id', research.id);
        const response = await fetch(`${import.meta.env.VITE_API_URL}/history/analyze/compare?${filters.toString()}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        const data = await response.json();
        const newComparisonData = [...comparisonData];
        newComparisonData[index] = data.comparison;
        setComparisonData(newComparisonData);
        setNetworkData(data.original);
    };

    const handleReset = (index) => {
        setFiltersData({
            nodeFilter: '',
            minWeight: 1,
            metrics: []
        });
        const newComparisonData = [...comparisonData];
        newComparisonData[index] = research.comparisons[index];
        setComparisonData(newComparisonData);
        setNetworkData({ nodes: research.analysis.nodes, links: research.analysis.links });
    };

    useEffect(() => {
        if (
            networkData &&
            networkData.nodes?.length > 0
        ) {
            fetchCommunityData();
        }
    }, []);


    useEffect(() => {
        networkData && calculateNetworkStats();
        if (forceGraphRef.current) {
            forceGraphRef.current.zoomToFit(400, 100);
        }
    }, [showMetrics, networkData]);

    return (
        <>
            <div className="d-flex justify-content-between align-items-center mb-4">
                <h3 className="fw-bold">{research.research_name}</h3>
                <div className="d-flex align-items-center gap-1">
                    {research.platform === 'whatsapp' ? (
                        <FaWhatsapp size={20} color="#25D366" />
                    ) : (
                        <FaWikipediaW size={20} color="#636466" />
                    )}
                    {/* <div>
                        <p className="m-0">Platform: {research.platform.charAt(0).toUpperCase() + research.platform.slice(1)}</p>
                    </div> */}
                </div>
            </div>

            <div className="mb-4">
                <h5 className="fw-bold">Description</h5>
                <p>{research.description || 'No description provided'}</p>
            </div>

            <div>
                <h5 className="fw-bold">Applied Filters</h5>
                <div className="row g-3 mt-3">
                    {filters.map(filter => (
                        filter.value && (
                            <div key={filter.label} className="col-12 col-md-6 col-lg-4">
                                <div className="p-3 bg-light rounded">
                                    <strong>{filter.label}:</strong> {filter.value}
                                </div>
                            </div>
                        )
                    ))}
                </div>
            </div>

            <div className="my-5">
                <h5 className='fw-bold'>Primary Analysis</h5>
                <div className='graph-container'>
                    <Row className="mt-4">
                        <Col
                            lg={3}
                            md={12}
                            className={`mb-3 metrics-panel ${showMetrics ? "open" : "closed"
                                }`}
                        >
                            <Card className="metrics-card">
                                <h5 className="fw-bold d-flex justify-content-between align-items-center">
                                    {showMetrics && "Graph Metrics"}
                                    <Button
                                        variant="link"
                                        className="metrics-toggle"
                                        onClick={() => setShowMetrics(!showMetrics)}
                                    >
                                        {showMetrics ? (
                                            <ChevronLeft size={20} />
                                        ) : (
                                            <ChevronRight size={20} />
                                        )}
                                    </Button>
                                </h5>
                                {showMetrics && (
                                    <div className="mt-2">
                                        <MetricsButton
                                            graphMetrics={graphMetrics}
                                            selectedMetric={selectedMetric}
                                            onToggleMetric={handleToggleMetric}
                                            onDensity={handleDensityMetric}
                                            onDiameter={handleDiameterMetric}
                                        />
                                        <Button
                                            className="metrics-item"
                                            onClick={() => setShowDataTable(!showDataTable)}
                                            variant={showDataTable ? "primary" : "outline-primary"}
                                        >
                                            <FileBarGraph className="me-1" /> Explore Data Table
                                        </Button>

                                        <Button
                                            className={`metrics-item ${strongConnectionsActive ? "active" : ""
                                                }`}
                                            onClick={handleStrongConnections}
                                        >
                                            {strongConnectionsActive
                                                ? "Show All Connections"
                                                : "Strongest Connections"}
                                        </Button>
                                        <Button
                                            className={`metrics-item ${highlightCentralNodes ? "active" : ""
                                                }`}
                                            onClick={handleHighlightCentralNodes}
                                        >
                                            Highlight Central Nodes
                                        </Button>
                                        <Button
                                            className={`metrics-item ${networkWasRestored ? "active" : ""
                                                }`}
                                            onClick={handleRestoreNetwork}
                                        >
                                            Restore Original Network
                                        </Button>
                                        <Button
                                            className={`metrics-item ${activityFilterEnabled === true ? "active" : ""
                                                }`}
                                            onClick={handleActivityFilter}
                                        >
                                            {activityFilterEnabled
                                                ? "Show All Users"
                                                : "Hide Inactive Users"}
                                        </Button>

                                        <Button
                                            className={`metrics-item ${showOnlyIntraCommunityLinks ? "active" : ""
                                                }`}
                                            onClick={handleToggleCommunitiesFilter}
                                        >
                                            {showOnlyIntraCommunityLinks
                                                ? "Show All Links"
                                                : "Hide Cross-Community Links"}
                                        </Button>
                                    </div>
                                )}
                            </Card>
                            <MetricsPanel networkStats={networkStats} />{" "}
                        </Col>

                        {/* Graph Display */}
                        <Col lg={9} md={12} className="graph-area">
                            {networkData && (
                                <NetworkCustomizationToolbar
                                    networkData={networkData}
                                    communities={communities}
                                    onApplyCustomization={handleNetworkCustomization}
                                    initialSettings={visualizationSettings}
                                />
                            )}
                            {(showDensity || showDiameter) && (
                                <Card className="density-card">
                                    {showDensity && (
                                        <h5 className="fw-bold">Graph Density: {densityValue}</h5>
                                    )}
                                    {showDiameter && (
                                        <h5 className="fw-bold">
                                            Graph Diameter: {diameterValue}
                                        </h5>
                                    )}
                                </Card>
                            )}

                            <Card className="graph-card">
                                <div className="graph-placeholder">
                                    {networkData && (
                                        <div className="graph-container">
                                            <NetworkGraph
                                                networkData={networkData}
                                                filteredNodes={filteredNodes}
                                                filteredLinks={filteredLinks}
                                                customizedNetworkData={customizedNetworkData}
                                                selectedMetric={selectedMetric}
                                                highlightCentralNodes={highlightCentralNodes}
                                                showMetrics={showMetrics}
                                                visualizationSettings={visualizationSettings}
                                                handleNodeClick={handleNodeClick}
                                                networkWasRestored={networkWasRestored}
                                                forceGraphRef={forceGraphRef}
                                            />
                                        </div>
                                    )}
                                </div>
                            </Card>
                            {showDataTable && networkData && (
                                <NetworkDataTable
                                    networkData={networkData}
                                    onClose={() => setShowDataTable(false)}
                                />
                            )}
                        </Col>
                    </Row>
                </div>
            </div>
        </>
    );
};

export default ResearchHistory;