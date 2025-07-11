import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { Container, Row, Col, Button, Card } from "react-bootstrap";
import {
  BsChevronLeft,
  BsChevronRight,
  BsFileBarGraph,
} from "react-icons/bs";
import { FaBalanceScale } from "react-icons/fa";
import { FiRefreshCw } from "react-icons/fi";
import { toast } from "sonner";
import { ForceGraph2D } from "react-force-graph";
import "./Home.css";
import { GraphContainer } from "./Form.style.js";
import NetworkCustomizationToolbar from "../components/NetworkCustomizationToolbar.jsx";
import ComparisonPanel from "../components/comparison/ComparisonPanel.jsx";
import ResearchCard from "../components/common/ResearchCard.jsx";
import MetricsButton from "../components/common/MetricsButton.jsx";
import ActivitySlider from "../components/common/ActivitySlider.jsx";
import NetworkDataTable from "../components/NetworkDataTable.jsx";
import useComparison from "../hooks/useComparison.jsx";
import useFilters from "../hooks/useFilters.jsx";
import FilterForm from "../components/filters/FilterForm.jsx";
import NetworkGraph from "../components/network/NetworkGraph.jsx";
import MetricsPanel from "../components/network/MetricsPanel.jsx";

import {
  uploadFile,
  deleteFile,
  saveFormToDB,
  analyzeNetwork,
  detectCommunities,
  compareNetworks,
  analyzeDecayingNetwork,
  saveToDB
} from "../components/utils/ApiService.js";
import { useDispatch, useSelector } from "react-redux";
import { addToMain, clearImages } from "../redux/images/imagesSlice.js";
import { FaCamera } from "react-icons/fa";
import Modal from "../components/utils/Modal.jsx";



export const graphMetrics = [
  "Degree Centrality",
  "Betweenness Centrality",
  "Closeness Centrality",
  "Eigenvector Centrality",
  "PageRank Centrality",
  "Density",
  "Diameter",
];

const Home = () => {
  const [showDownload, setShowDownload] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);
  const [message, setMessage] = useState("");
  const [uploadedFile, setUploadedFile] = useState("");
  const [chartData, setChartData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [inputKey, setInputKey] = useState(Date.now());
  const [showMetrics, setShowMetrics] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [showDensity, setShowDensity] = useState(false);
  const [densityValue, setDensityValue] = useState(0);
  const [showDiameter, setShowDiameter] = useState(false);
  const [diameterValue, setDiameterValue] = useState(0);
  const [showNetworkStats, setShowNetworkStats] = useState(false);
  const [networkStats, setNetworkStats] = useState({});
  const [originalNetworkData, setOriginalNetworkData] = useState(null);
  const [strongConnectionsActive, setStrongConnectionsActive] = useState(false);
  const [communities, setCommunities] = useState([]);
  const [customizedNetworkData, setCustomizedNetworkData] = useState(null);
  const [highlightCentralNodes, setHighlightCentralNodes] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showDataTable, setShowDataTable] = useState(false);
  const [showRemoveNodeModal, setShowRemoveNodeModal] = useState(false);
  const [activityFilterEnabled, setActivityFilterEnabled] = useState(false);
  const [activityThreshold, setActivityThreshold] = useState(2);
  const forceGraphRef = useRef(null);
  const [networkWasRestored, setNetworkWasRestored] = useState(false);
  const [shouldFetchCommunities, setShouldFetchCommunities] = useState(false);
  const [nodesFixed, setNodesFixed] = useState(false);
  const [showOnlyIntraCommunityLinks, setShowOnlyIntraCommunityLinks] =
    useState(false);
  const [isDirectedGraph, setIsDirectedGraph] = useState(false);
  const [openModal, setOpenModal] = useState(false);
  const [normalized, setNormalized] = useState({
    isNormalized: false,
    data: null
  });
  const [userWeightSums, setUserWeightSums] = useState({});


  const [communityMap, setCommunityMap] = useState({});
  const {
    comparisonCount,
    comparisonFiles,
    comparisonData,
    activeComparisonIndices,
    comparisonNetworkData,
    filteredOriginalData,
    filteredComparisonData,
    comparisonFilter,
    minComparisonWeight,
    comparisonMetrics,
    highlightCommonNodes,
    addComparison,
    handleComparisonFileChange,
    toggleComparisonActive,
    analyzeComparisonNetwork,
    applyComparisonFilters,
    resetComparisonFilters,
    calculateComparisonStats,
  } = useComparison(originalNetworkData, uploadedFile);

  const filters = useFilters();
  const {
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    startTime,
    setStartTime,
    endTime,
    setEndTime,
    messageLimit,
    setMessageLimit,
    limitType,
    setLimitType,
    minMessageLength,
    setMinMessageLength,
    maxMessageLength,
    setMaxMessageLength,
    keywords,
    setKeywords,
    usernameFilter,
    setUsernameFilter,
    minMessages,
    setMinMessages,
    maxMessages,
    setMaxMessages,
    activeUsers,
    setActiveUsers,
    selectedUsers,
    setSelectedUsers,
    weightCalculationDepth,
    setWeightCalculationDepth,
    isAnonymized,
    setIsAnonymized,
    showFilters,
    setShowFilters,
    filter,
    setFilter,
    buildNetworkFilterParams,
    formatTime,
    resetFilters,
    handleInputChange,
  } = filters;
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
  const dispatch = useDispatch();

  const unfixAllNodes = () => {
    if (networkData && networkData.nodes) {
      const hasFixedNodes = networkData.nodes.some(
        (node) => node.fx !== null || node.fy !== null
      );

      if (hasFixedNodes) {
        const updatedNodes = networkData.nodes.map((node) => ({
          ...node,
          fx: null,
          fy: null,
        }));

        setNetworkData({
          ...networkData,
          nodes: updatedNodes,
        });

        if (forceGraphRef.current) {
          forceGraphRef.current.d3ReheatSimulation();
        }

        setNodesFixed(false);
      }
    }
  };

  const { currentUser } = useSelector((state) => state.user);

  const loadCommunityColors = (communities) => {
    if (!communities || communities.length === 0) {
      console.log("No communities to set colors for");
      return;
    }

    console.log("Setting up colors for", communities.length, "communities");

    const communityColors = {};
    const communityNames = {};

    communities.forEach((community, index) => {
      const communityId = community.id;
      const colorArray = visualizationSettings.customColors.communityColors;
      communityColors[communityId] = colorArray[index % colorArray.length];
      communityNames[communityId] = `Community ${communityId}`;
      console.log(
        `Community ${communityId} gets color ${communityColors[communityId]}`
      );
    });

    const updatedSettings = {
      ...visualizationSettings,
      communityColors: communityColors,
      communityNames: communityNames,
      colorBy: "community",
      highlightCommunities: [],
    };

    setVisualizationSettings(updatedSettings);
    console.log("Updated visualization settings:", updatedSettings);

    handleNetworkCustomization(updatedSettings);
  };

  useEffect(() => {
    if (!uploadedFile) {
      setFile(null);
      setChartData(null);
      setNetworkData(null);
      filters.setFilter("");
      filters.setStartDate("");
      filters.setEndDate("");
      filters.setMessageLimit(50);
      filters.setKeywords("");
      setInputKey(Date.now());
      if (forceGraphRef.current) {
        forceGraphRef.current.zoomToFit(400, 100);
      }
    }
    if (networkData) {
      calculateNetworkStats();
      networkData.links && !normalized.isNormalized && setUserWeightSums(
        networkData.links.reduce((acc, link) => {
          acc[link.target] = (acc[link.target] || 0) + link.weight;
          return acc;
        }, {})
      );
    }
    if (
      shouldFetchCommunities &&
      networkData &&
      networkData.nodes?.length > 0
    ) {
      fetchCommunityData();
      setShouldFetchCommunities(false);
    }
  }, [shouldFetchCommunities, uploadedFile, showMetrics, networkData]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    const newFileName = `${selectedFile.name.split(".")[0]}-${Date.now()}.${selectedFile.name.split(".")[1]}`;
    const renamedFile = new File([selectedFile], newFileName, { type: selectedFile.type });

    setFile(renamedFile);
    setUploadedFile("");
    setChartData(null);
    setNetworkData(null);
    setMessage("");
    handleSubmit(renamedFile);
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = (selectedFile) => {
    if (!selectedFile) {
      toast.error("Please select a file before uploading.");
      return;
    }

    toast.promise(
      uploadFile(selectedFile),
      {
        loading: "Uploading...",
        success: (data) => {
          if (data.message) {
            setUploadedFile(data.filename);
          }
          return data?.message || "File uploaded successfully!";
        },
        error: (error) => {
          return error?.message || "Error uploading file.";
        },
      }
    );
  };

  const handleDelete = async () => {
    if (!uploadedFile) {
      toast.error("No file selected to delete.");
      return;
    }

    toast.promise(
      deleteFile(uploadedFile),
      {
        loading: "Deleting...",
        success: (data) => {
          if (data.success) {
            setMessage(data.message || "File deleted successfully!");
            setUploadedFile("");
            setFile(null);
            setChartData(null);
            setNetworkData(null);
            filters.setFilter("");
            filters.setStartDate("");
            filters.setEndDate("");
            filters.setMessageLimit(50);
            filters.setKeywords("");
            setInputKey(Date.now());
          } else {
            toast.error("Error: Could not delete the file.");
          }
          return data?.message || "File deleted successfully!";
        },
        error: (error) => {
          console.error("Error deleting file:", error);
          return error?.message || "Error deleting file.";
        },
      }
    );

  };

  const handleNetworkAnalysis = () => {
    if (!uploadedFile) {
      toast.error("No file selected for analysis.");
      return;
    }

    setNetworkWasRestored(false);

    const params = filters.buildNetworkFilterParams();

    const correctFunction = isDirectedGraph ? analyzeDecayingNetwork : analyzeNetwork;

    toast.promise(
      correctFunction(uploadedFile, params),
      {
        loading: "Analyzing network...",
        success: (data) => {
          if (data.nodes && data.links) {
            dispatch(clearImages());
            setNetworkData(data);
            setOriginalNetworkData(data);
            setShouldFetchCommunities(true);
            isDirectedGraph && setNormalized({ isNormalized: false, data: null });
            return "Analysis completed successfully!";
          } else {
            return "No data returned from server.";
          }
        },
        error: (error) => {
          return error?.message || "Error analyzing network.";
        },
      }
    )
  };

  const getSumWeightOfUser = (user) => {
    return userWeightSums[user] || 0;
  };

  const handleNormalizeGraphData = (e) => {
    e.stopPropagation()
    setNetworkData(data => {
      const newData = {
        ...data,
        links: normalized.isNormalized ? normalized.data.map((n) => ({ ...n })) : data.links.map((n) => ({ ...n, weight: +(n.weight / getSumWeightOfUser(n.target)).toFixed(1) }))
      };
      normalized.isNormalized ? setNormalized({
        isNormalized: false,
        data: null
      }) : setNormalized({
        isNormalized: true,
        data: data.links.map((n) => ({ ...n }))
      });
      return newData;
    });
    setOriginalNetworkData(data => {
      const newData = {
        ...data,
        links: normalized.isNormalized ? normalized.data.map((n) => ({ ...n })) : data.links.map((n) => ({ ...n, weight: 1 }))
      };
      return newData;
    });
  };

  const handleSaveToDB = () => {
    const params = filters.buildNetworkFilterParams();
    const id = currentUser?.id;
    if (!name || !description || !uploadedFile || !params || !id) {
      if (!id) { setOpenModal(true); return; }
      let msg = "Please fill in all required fields.(";
      if (!name) msg += " Name is required.";
      if (!description) msg += " Description is required.";
      if (!uploadedFile) msg += " File is required.";
      if (!params) msg += " Filters are required.";
      msg += ")";
      toast.error(msg);
      return;
    }
    toast.promise(
      saveToDB(id, name, description, uploadedFile, params, selectedMetric, {
        hasComparison: comparisonNetworkData.length ? true : false,
        data: comparisonNetworkData || undefined,
      }),
      {
        loading: "Saving...",
        success: (data) => {
          return data?.message || "Research saved successfully!";
        },
        error: (error) => {
          return error?.message || "Error saving research.";
        },
      }
    )
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

  const handleToggleMetric = (metric) => {
    setSelectedMetric(selectedMetric === metric ? null : metric);
  };

  const handleDensityMetric = () => {
    const density = calculateDensity(networkData.nodes, networkData.links);
    setDensityValue(density.toFixed(4));
    setShowDensity(!showDensity);
  };

  const calculateDensity = (nodes, links) => {
    const n = nodes.length;
    const m = links.length;
    if (n < 2) return 0;
    return (2 * m) / (n * (n - 1));
  };

  const calculateDiameter = (nodes, links) => {
    if (nodes.length < 2) return 0;

    const distances = {};
    nodes.forEach((node) => {
      distances[node.id] = {};
      nodes.forEach(
        (n) => (distances[node.id][n.id] = node.id === n.id ? 0 : Infinity)
      );
    });

    links.forEach((link) => {
      distances[link.source][link.target] = 1;
      distances[link.target][link.source] = 1;
    });

    // Floyd-Warshall
    nodes.forEach((k) => {
      nodes.forEach((i) => {
        nodes.forEach((j) => {
          if (
            distances[i.id][j.id] >
            distances[i.id][k.id] + distances[k.id][j.id]
          ) {
            distances[i.id][j.id] =
              distances[i.id][k.id] + distances[k.id][j.id];
          }
        });
      });
    });

    let maxDistance = 0;
    nodes.forEach((i) => {
      nodes.forEach((j) => {
        if (distances[i.id][j.id] !== Infinity) {
          maxDistance = Math.max(maxDistance, distances[i.id][j.id]);
        }
      });
    });

    return maxDistance;
  };

  const handleDiameterMetric = () => {
    setShowDiameter(!showDiameter);
    if (!showDiameter && networkData) {
      const diameter = calculateDiameter(networkData.nodes, networkData.links);
      setDiameterValue(diameter);
    }
  };

  const filteredNodes = networkData
    ? networkData.nodes.filter((node) =>
      node.id.toLowerCase().includes(filter.toLowerCase())
    )
    : [];


  const filteredLinks = networkData
    ? networkData.links.filter(
      (link) =>
        filteredNodes.some((node) => node.id === link.source) &&
        filteredNodes.some((node) => node.id === link.target)
    )
    : [];

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

  const handleComparisonAnalysis = (index) => {
    const comparisonFile = comparisonData[index];
    if (!comparisonFile || !comparisonFile.filename) {
      setMessage("Please select a comparison file first.");
      return;
    }

    const params = buildNetworkFilterParams();

    analyzeNetwork(comparisonFile.filename, params)
      .then((data) => {
        console.log(`Comparison data ${index} returned:`, data);
        if (data.nodes && data.links) {
          const updatedComparisonData = [...comparisonNetworkData];
          updatedComparisonData[index] = data;
          setComparisonNetworkData(updatedComparisonData);

          if (!activeComparisonIndices.includes(index)) {
            setActiveComparisonIndices([...activeComparisonIndices, index]);
          }

          setMessage(
            `Comparison analysis ${index + 1} completed successfully!`
          );
        } else {
          setMessage(`No valid data returned for comparison ${index + 1}.`);
        }
      })
      .catch((error) => {
        setMessage(`Error analyzing comparison file ${index + 1}.`);
        console.error(`Error during comparison analysis ${index}:`, error);
      });
  };

  const renderComparisonGraph = (index) => {
    const compData = comparisonNetworkData[index];
    if (!compData || !compData.nodes || !compData.links) {
      return <div>No data available for this comparison</div>;
    }

    const filteredNodes = compData.nodes.filter((node) =>
      node.id.toLowerCase().includes(filter.toLowerCase())
    );

    const filteredLinks = compData.links.filter(
      (link) =>
        filteredNodes.some(
          (node) =>
            node.id === link.source ||
            (typeof link.source === "object" && node.id === link.source.id)
        ) &&
        filteredNodes.some(
          (node) =>
            node.id === link.target ||
            (typeof link.target === "object" && node.id === link.target.id)
        )
    );
  };

  const fetchCommunityData = () => {
    if (!uploadedFile) {
      toast.error("No file selected for community detection.");
      return;
    }

    const params = filters.buildNetworkFilterParams();


    toast.promise(
      detectCommunities(uploadedFile, params),
      {
        loading: "Detecting communities...",
        success: (data) => {

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
                  console.log(
                    `Assigning node ${node.id} to community ${community}`
                  );
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

              toast.success(
                `Detected ${data.communities.length} communities in the network.`
              );
            }
          } else {
            toast.error("No community data returned from server.");
          }
        },
        error: (error) => {
          return error?.message || "Error detecting communities.";
        },
      }
    )
  };

  const detectAndApplyCommunityData = () => {
    if (!uploadedFile) {
      toast.error("No file selected for community detection.");
      return;
    }

    const params = filters.buildNetworkFilterParams();

    toast.promise(
      detectCommunities(uploadedFile, params),
      {
        loading: "Detecting communities...",
        success: (data) => {
          if (data.communities && data.nodes) {
            setCommunities(data.communities);

            const newCommunityMap = {};

            data.nodes.forEach((node) => {
              if (node.community !== undefined) {
                newCommunityMap[node.id.toString().trim()] = node.community;
              }
            });

            console.log("CommunityMap:", newCommunityMap);
            setCommunityMap(newCommunityMap);

            if (networkData && networkData.nodes) {
              const updatedNodes = networkData.nodes.map((node) => {
                const normalizedId = node.id.toString().trim();
                const community = newCommunityMap[normalizedId];

                if (community !== undefined) {
                  console.log(
                    `Assigning node ${node.id} to community ${community}`
                  );
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

              toast.success(
                `Detected ${data.communities.length} communities in the network.`
              );
            }
          } else {
            toast.error("No community data returned from server.");
          }
        },
        error: (error) => {
          return error?.message || "Error detecting communities.";
        },
      }
    );
  };

  const handleNetworkCustomization = (settings) => {
    setVisualizationSettings(settings);
    console.log("Applying visualization settings:", settings);

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

  const interpolateColor = (color1, color2, ratio) => {
    const r1 = parseInt(color1.substring(1, 3), 16);
    const g1 = parseInt(color1.substring(3, 5), 16);
    const b1 = parseInt(color1.substring(5, 7), 16);

    const r2 = parseInt(color2.substring(1, 3), 16);
    const g2 = parseInt(color2.substring(3, 5), 16);
    const b2 = parseInt(color2.substring(5, 7), 16);

    const r = Math.round(r1 + (r2 - r1) * ratio);
    const g = Math.round(g1 + (g2 - g1) * ratio);
    const b = Math.round(b1 + (b2 - b1) * ratio);

    return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
  };

  const standardizeGraphData = (graphData) => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      return graphData;
    }

    const nodeMap = {};
    graphData.nodes.forEach((node) => {
      nodeMap[node.id] = node;
    });

    const standardizedLinks = graphData.links.map((link) => {
      const sourceId =
        typeof link.source === "object" ? link.source.id : link.source;
      const targetId =
        typeof link.target === "object" ? link.target.id : link.target;

      if (!nodeMap[sourceId] || !nodeMap[targetId]) {
        console.warn(`Invalid link: source=${sourceId}, target=${targetId}`);
      }

      return {
        source: sourceId,
        target: targetId,
        weight: link.weight || 1,
      };
    });

    return {
      nodes: graphData.nodes,
      links: standardizedLinks,
    };
  };

  const renderForceGraph = (
    graphData,
    width,
    height,
    isComparisonGraph = false
  ) => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      return <div>No data available</div>;
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

    const baseColor = isComparisonGraph ? "purple" : "blue";
    const linkColor = isComparisonGraph
      ? "rgba(128, 0, 128, 0.6)"
      : "rgba(128, 128, 128, 0.6)";

    const getNodeSize = (node) => {
      if (!node) return 20;

      if (filteredOriginalData && filteredComparisonData) {
        if (comparisonMetrics.includes("Degree Centrality"))
          return Math.max(10, node.degree * 80);
        if (comparisonMetrics.includes("Betweenness Centrality"))
          return Math.max(10, node.betweenness * 80);
        if (comparisonMetrics.includes("Closeness Centrality"))
          return Math.max(10, node.closeness * 50);
        if (comparisonMetrics.includes("Eigenvector Centrality"))
          return Math.max(10, node.eigenvector * 60);
        if (comparisonMetrics.includes("PageRank Centrality"))
          return Math.max(10, node.pagerank * 500);
      } else if (!isComparisonGraph && selectedMetric) {
        if (selectedMetric === "Degree Centrality")
          return Math.max(10, node.degree * 80);
        if (selectedMetric === "Betweenness Centrality")
          return Math.max(10, node.betweenness * 80);
        if (selectedMetric === "Closeness Centrality")
          return Math.max(10, node.closeness * 50);
        if (selectedMetric === "Eigenvector Centrality")
          return Math.max(10, node.eigenvector * 60);
        if (selectedMetric === "PageRank Centrality")
          return Math.max(10, node.pagerank * 500);
      }

      return 20;
    };

    return (
      <ForceGraph2D
        graphData={processedData}
        width={width}
        height={height}
        fitView
        fitViewPadding={20}
        nodeAutoColorBy="id"
        linkWidth={(link) => Math.sqrt(link.weight || 1)}
        linkColor={() => linkColor}
        directed={isDirectedGraph}
        onNodeDragEnd={(node) => {
          node.fx = node.x;
          node.fy = node.y;
          setNodesFixed(true);
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const fontSize = 12 / globalScale;

          const radius =
            node.size ||
            (selectedMetric === "PageRank Centrality"
              ? Math.max(10, node.pagerank * 500)
              : selectedMetric === "Eigenvector Centrality"
                ? Math.max(10, node.eigenvector * 60)
                : selectedMetric === "Closeness Centrality"
                  ? Math.max(10, node.closeness * 50)
                  : selectedMetric === "Betweenness Centrality"
                    ? Math.max(10, node.betweenness * 80)
                    : selectedMetric === "Degree Centrality"
                      ? Math.max(10, node.degree * 80)
                      : 20);

          ctx.save();
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);

          const isCommunityHighlighted =
            node.community !== undefined &&
            visualizationSettings.highlightCommunities &&
            visualizationSettings.highlightCommunities.includes(
              parseInt(node.community, 10)
            );

          const isNodeHighlighted = node.highlighted && highlightCentralNodes;

          let nodeColor;

          if (node.color) {
            nodeColor = node.color;
          } else if (selectedMetric === "PageRank Centrality") {
            nodeColor = "orange";
          } else if (selectedMetric === "Eigenvector Centrality") {
            nodeColor = "purple";
          } else if (selectedMetric === "Closeness Centrality") {
            nodeColor = "green";
          } else if (selectedMetric === "Betweenness Centrality") {
            nodeColor = "red";
          } else if (selectedMetric === "Degree Centrality") {
            nodeColor = "#231d81";
          } else {
            nodeColor = "blue";
          }
          if (node.isHighlightedCommunity && node.community !== undefined) {
            const communityName =
              visualizationSettings.communityNames?.[node.community] ||
              `Community ${node.community}`;
            ctx.fillStyle = "#F5BD20";
            ctx.font = `${fontSize * 0.8}px Sans-Serif`;
            ctx.fillText(communityName, node.x, node.y + radius + 15);
          }
          ctx.fillStyle = nodeColor;
          ctx.fill();

          if (isCommunityHighlighted) {
            ctx.strokeStyle =
              visualizationSettings.customColors.highlightNodeColor;
            ctx.lineWidth = 3;
            ctx.stroke();

            console.log(
              `Rendering highlighted community node: ${node.id}, community: ${node.community}, color: ${nodeColor}`
            );
          } else if (isNodeHighlighted) {
            ctx.strokeStyle = "#ffff00";
            ctx.lineWidth = 3;
            ctx.stroke();
          }

          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = "white";
          ctx.fillText(node.id, node.x, node.y);

          if (selectedMetric === "Degree Centrality") {
            ctx.fillStyle = "#231d81";
            ctx.fillText(`Deg: ${node.degree}`, node.x, node.y + radius + 5);
          }
          if (selectedMetric === "Betweenness Centrality") {
            ctx.fillStyle = "DarkRed";
            ctx.fillText(
              `Btw: ${node.betweenness?.toFixed(2) || 0}`,
              node.x,
              node.y + radius + 5
            );
          }
          if (selectedMetric === "Closeness Centrality") {
            ctx.fillStyle = "green";
            ctx.fillText(
              `Cls: ${node.closeness?.toFixed(2) || 0}`,
              node.x,
              node.y + radius + 5
            );
          }
          if (selectedMetric === "Eigenvector Centrality") {
            ctx.fillStyle = "purple";
            ctx.fillText(
              `Eig: ${node.eigenvector?.toFixed(4) || 0}`,
              node.x,
              node.y + radius + 5
            );
          }
          if (selectedMetric === "PageRank Centrality") {
            ctx.fillStyle = "orange";
            ctx.fillText(
              `PR: ${node.pagerank?.toFixed(4) || 0}`,
              node.x,
              node.y + radius + 5
            );
          }
          ctx.restore();
        }}
      />
    );
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

  const handleNodeClick = (node) => {
    if (networkWasRestored) {
      setSelectedNode(node);
      setShowRemoveNodeModal(true);
    }
  };

  const handleRemoveNode = () => {
    if (!selectedNode || !networkData) return;

    const updatedNodes = networkData.nodes.filter(
      (node) => node.id !== selectedNode.id
    );

    const updatedLinks = networkData.links.filter(
      (link) =>
        link.source !== selectedNode.id &&
        link.target !== selectedNode.id &&
        link.source.id !== selectedNode.id &&
        link.target.id !== selectedNode.id
    );

    if (!originalNetworkData) {
      setOriginalNetworkData(JSON.parse(JSON.stringify(networkData)));
    }

    setNetworkData({
      nodes: updatedNodes,
      links: updatedLinks,
    });

    setShowRemoveNodeModal(false);
    setSelectedNode(null);
  };

  const handleRestoreNetwork = () => {
    if (originalNetworkData) {
      if (networkWasRestored) {
        setNetworkWasRestored(false);
      } else {
        setNetworkData(JSON.parse(JSON.stringify(originalNetworkData)));
        setNetworkWasRestored(true);

        if (activityFilterEnabled) {
          setActivityFilterEnabled(false);
        }
      }
    }
  };

  const handleActivityFilter = () => {
    if (!networkData) return;

    const newState = !activityFilterEnabled;
    setActivityFilterEnabled(newState);

    if (newState) {
      if (!originalNetworkData) {
        setOriginalNetworkData(JSON.parse(JSON.stringify(networkData)));
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
        setNetworkData(JSON.parse(JSON.stringify(originalNetworkData)));
        setNetworkWasRestored(true);
      }
    }
  };

  const applyActivityFilter = (threshold) => {
    if (!networkData) return;
    if (!originalNetworkData && !activityFilterEnabled) {
      setOriginalNetworkData(JSON.parse(JSON.stringify(networkData)));
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
      (node) => connectionCounts[node.id] >= threshold
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
  };

  const handleToggleCommunitiesFilter = () => {
    if (!networkData || !originalNetworkData) return;

    if (!communityMap || Object.keys(communityMap).length === 0) {
      toast.error(
        "Community data not found. Detecting communities..."
      );
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

      console.log("Updated nodes:", updatedNodes);
      console.log("Filtered links:", filteredLinks);

      setNetworkData({
        nodes: updatedNodes,
        links: filteredLinks,
      });

      setMessage(
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

      setMessage("Showing all links in the network.");
    }

    if (forceGraphRef.current) {
      setTimeout(() => {
        forceGraphRef.current.d3ReheatSimulation();
        forceGraphRef.current.zoomToFit(400);
      }, 100);
    }
  };

  const handleScreenshot = (e, index) => {
    e.stopPropagation();
    const canvas = document.querySelectorAll('canvas');
    canvas.length > index && dispatch(addToMain({
      data: canvas[index].toDataURL("image/png"),
    }));
  }

  return (
    <Container fluid className="upload-section">
      {
        openModal && (
          <Modal>
            <div className="text-center p-4">
              <h5 className="mb-3">You need to sign in to save your research!</h5>
              <p className="mb-4">
                Please <Link to="/sign-in">sign in</Link> to continue.
              </p>
              <Button variant="primary" onClick={() => setOpenModal(false)}>
                Close
              </Button>
            </div>
          </Modal>
        )
      }
      <ResearchCard
        filters={filters}
        name={name}
        setName={setName}
        description={description}
        setDescription={setDescription}
        isAnonymized={isAnonymized}
        setIsAnonymized={setIsAnonymized}
        handleUploadClick={handleUploadClick}
        handleDelete={handleDelete}
        fileInputRef={fileInputRef}
        handleFileChange={handleFileChange}
        inputKey={inputKey}
        message={message}
        handleSave={handleSaveToDB}
        showDownload={showDownload}
        setShowDownload={setShowDownload}
        networkData={networkData ? true : false}
        selectedMetric={selectedMetric}
        hasComparison={comparisonNetworkData.length ? true : false}
        isDirectedGraph={isDirectedGraph}
        setIsDirectedGraph={setIsDirectedGraph}
        weightCalculationDepth={weightCalculationDepth}
        setWeightCalculationDepth={setWeightCalculationDepth}
      />

      {uploadedFile && (
        <div>
          <div>
            <FilterForm
              startDate={startDate}
              setStartDate={setStartDate}
              endDate={endDate}
              setEndDate={setEndDate}
              startTime={startTime}
              setStartTime={setStartTime}
              endTime={endTime}
              setEndTime={setEndTime}
              messageLimit={messageLimit}
              setMessageLimit={setMessageLimit}
              limitType={limitType}
              setLimitType={setLimitType}
              minMessageLength={minMessageLength}
              setMinMessageLength={setMinMessageLength}
              maxMessageLength={maxMessageLength}
              setMaxMessageLength={setMaxMessageLength}
              keywords={keywords}
              setKeywords={setKeywords}
              usernameFilter={usernameFilter}
              setUsernameFilter={setUsernameFilter}
              minMessages={minMessages}
              setMinMessages={setMinMessages}
              maxMessages={maxMessages}
              setMaxMessages={setMaxMessages}
              activeUsers={activeUsers}
              setActiveUsers={setActiveUsers}
              selectedUsers={selectedUsers}
              setSelectedUsers={setSelectedUsers}
              isAnonymized={isAnonymized}
              setIsAnonymized={setIsAnonymized}
              handleNetworkAnalysis={handleNetworkAnalysis}
            />
          </div>

          {activityFilterEnabled && (
            <ActivitySlider
              activityThreshold={activityThreshold}
              setActivityThreshold={setActivityThreshold}
              applyActivityFilter={applyActivityFilter}
            />
          )}

          {uploadedFile && (
            <Row className="mt-4">
              <Col
                lg={3}
                md={12}
                className={`mb-3 metrics-panel ${showMetrics ? "open" : "closed"
                  }`}
              >
                <Card className="metrics-card">
                  <h4 className="fw-bold d-flex justify-content-between align-items-center">
                    {showMetrics && "Graph Metrics"}
                    <Button
                      variant="link"
                      className="metrics-toggle"
                      onClick={() => setShowMetrics(!showMetrics)}
                    >
                      {showMetrics ? (
                        <BsChevronLeft size={20} />
                      ) : (
                        <BsChevronRight size={20} />
                      )}
                    </Button>
                  </h4>
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
                        <BsFileBarGraph className="me-1" /> Explore Data Table
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
                      <Button
                        className={`metrics-item ${nodesFixed ? "active" : ""}`}
                        onClick={unfixAllNodes}
                      >
                        Release All Nodes
                      </Button>
                      <Button
                        className={`metrics-item ${isDirectedGraph ? "active" : ""
                          }`}
                        onClick={() => setIsDirectedGraph(!isDirectedGraph)}
                      >
                        {isDirectedGraph
                          ? "Show Undirected Graph"
                          : "Show Directed Graph"}
                      </Button>
                    </div>
                  )}
                </Card>
                <MetricsPanel networkStats={networkStats} />
              </Col>

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
                      <GraphContainer>
                        {isDirectedGraph && <button
                          className="graph-button-2"
                          data-tooltip-id="my-tooltip"
                          data-tooltip-content={normalized.isNormalized ? "Restore Original Weights" : "Normalize Weights"}
                          data-tooltip-place="top"
                          onClick={handleNormalizeGraphData}>
                          {normalized.isNormalized ? <FiRefreshCw /> : <FaBalanceScale />}
                        </button>}
                        <button
                          className="graph-button"
                          data-tooltip-id="my-tooltip"
                          data-tooltip-content="Add screenshot to report"
                          data-tooltip-place="top"
                          onClick={(e) => handleScreenshot(e, 0)}><FaCamera /></button>
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
                          isDirectedGraph={isDirectedGraph}
                        />
                      </GraphContainer>
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
          )}

          <Row className="mt-4">
            <ComparisonPanel
              originalNetworkData={originalNetworkData}
              comparisonFiles={comparisonFiles}
              comparisonData={comparisonData}
              comparisonNetworkData={comparisonNetworkData}
              activeComparisonIndices={activeComparisonIndices}
              filteredOriginalData={filteredOriginalData}
              filteredComparisonData={filteredComparisonData}
              onFileUpload={handleComparisonFileChange}
              onAnalyzeNetwork={(index) => {
                analyzeComparisonNetwork(
                  index,
                  buildNetworkFilterParams()
                ).then((result) => {
                  if (result.message) {
                    toast.success(result.message);
                  }
                });
              }}
              onToggleComparison={toggleComparisonActive}
              onApplyComparisonFilters={(filters) => {
                const filtersWithNetworkParams = {
                  ...filters,
                  networkFilterParams: buildNetworkFilterParams(),
                };

                applyComparisonFilters(filtersWithNetworkParams).then(
                  (result) => {
                    if (result.message) {
                      toast.success(result.message);
                    }
                  }
                );
              }}
              onResetComparisonFilters={resetComparisonFilters}
              addComparison={addComparison}
              comparisonCount={comparisonCount}
            />
          </Row>
        </div>
      )}

      {showRemoveNodeModal && selectedNode && (
        <div
          className="modal show d-block"
          tabIndex="-1"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
        >
          <div className="modal-dialog">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">Remove Node from Network</h5>
                <button
                  type="button"
                  className="btn-close"
                  onClick={() => setShowRemoveNodeModal(false)}
                  aria-label="Close"
                ></button>
              </div>
              <div className="modal-body">
                <p>
                  Are you sure you want to remove the node
                  <strong>"{selectedNode.id}"</strong> from the network?
                </p>
                <p>
                  This will help analyze the impact of this user on the network
                  structure.
                </p>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowRemoveNodeModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleRemoveNode}
                >
                  Remove Node
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Container>
  );
};
export default Home;