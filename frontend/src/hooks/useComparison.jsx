import { useState, useEffect } from "react";
import { BASE_URL } from "../components/utils/ApiService.js";

const useComparison = (originalNetworkData, uploadedFile) => {
  const [comparisonCount, setComparisonCount] = useState(1);
  const [comparisonFiles, setComparisonFiles] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [activeComparisonIndices, setActiveComparisonIndices] = useState([]);
  const [comparisonNetworkData, setComparisonNetworkData] = useState([]);
  const [filteredOriginalData, setFilteredOriginalData] = useState(null);
  const [filteredComparisonData, setFilteredComparisonData] = useState({});
  const [comparisonFilter, setComparisonFilter] = useState("");
  const [minComparisonWeight, setMinComparisonWeight] = useState(1);
  const [comparisonMetrics, setComparisonMetrics] = useState([]);
  const [highlightCommonNodes, setHighlightCommonNodes] = useState(false);

  const [comparisonFilterSettings, setComparisonFilterSettings] = useState([]);

  useEffect(() => {
    if (!uploadedFile) {
      setComparisonNetworkData([]);
      setComparisonData([]);
      setComparisonFiles([]);
      setComparisonCount(1);
      setActiveComparisonIndices([]);
      setFilteredOriginalData(null);
      setFilteredComparisonData({});
      setComparisonFilterSettings([]);
    }
  }, [uploadedFile]);

  const getDefaultFilterSettings = () => ({
    timeFrame: {
      startDate: "",
      endDate: "",
      startTime: "",
      endTime: "",
    },
    limit: {
      enabled: false,
      count: 50,
      fromEnd: false,
      type: "messages",
    },
    messageCriteria: {
      minLength: 1,
      maxLength: "",
      keywords: "",
      contentFilter: "",
    },
    userFilters: {
      minMessages: 1,
      maxMessages: "",
      usernameFilter: "",
    },
    config: {
      directed: false,
      anonymize: false,
      history: false,
      messageCount: 3,
      normalized: false,
      messageWeights: [0.2, 0.3, 0.5],
    },
  });

  const addComparison = () => {
    setComparisonCount((prevCount) => prevCount + 1);
    setComparisonFilterSettings((prev) => [
      ...prev,
      getDefaultFilterSettings(),
    ]);
  };

  const updateComparisonFilterSettings = (index, newSettings) => {
    setComparisonFilterSettings((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...(updated[index] || getDefaultFilterSettings()),
        ...newSettings,
      };
      return updated;
    });
  };

  const handleComparisonFileChange = async (event, index, platform) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
  
    const updatedFiles = [...comparisonFiles];
    updatedFiles[index] = selectedFile;
    setComparisonFiles(updatedFiles);
  
    if (!comparisonFilterSettings[index]) {
      updateComparisonFilterSettings(index, getDefaultFilterSettings());
    }
  
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("platform", platform); 
  
    try {
      const response = await fetch(`${BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || response.statusText);
      }
  
      const data = await response.json();
  
      if (data.filename) {
        const updatedData = [...comparisonData];
        updatedData[index] = {
          id: index,
          filename: data.filename,
          name: selectedFile.name,
          isWikipediaData: platform === "wikipedia",
          isOriginalFile: false,
        };
        setComparisonData(updatedData);
        console.log(`Comparison file ${index} uploaded: ${data.filename}`);
        return {
          success: true,
          message: `File "${selectedFile.name}" uploaded successfully`,
        };
      }
    } catch (error) {
      console.error("Error uploading comparison file:", error);
      return {
        success: false,
        message: "An error occurred during comparison file upload.",
      };
    }
  };
  

  const toggleComparisonActive = (index) => {
    setActiveComparisonIndices((prev) => {
      if (prev.includes(index)) {
        return prev.filter((i) => i !== index);
      } else {
        return [...prev, index];
      }
    });
  };

  const buildFilterParamsFromSettings = (filterSettings) => {
    const params = new URLSearchParams();

    if (filterSettings.timeFrame) {
      if (filterSettings.timeFrame.startDate)
        params.append("start_date", filterSettings.timeFrame.startDate);
      if (filterSettings.timeFrame.endDate)
        params.append("end_date", filterSettings.timeFrame.endDate);
      if (filterSettings.timeFrame.startTime)
        params.append("start_time", filterSettings.timeFrame.startTime);
      if (filterSettings.timeFrame.endTime)
        params.append("end_time", filterSettings.timeFrame.endTime);
    }

    if (filterSettings.limit && filterSettings.limit.enabled) {
      params.append("limit", filterSettings.limit.count.toString());
      params.append(
        "limit_type",
        filterSettings.limit.fromEnd ? "last" : "first"
      );
    }
    if (filterSettings.config) {
      if (filterSettings.config.directed)
        params.append(
          "directed",
          filterSettings.config.directed ? "true" : "false"
        );
      if (filterSettings.config.anonymize)
        params.append(
          "anonymize",
          filterSettings.config.anonymize ? "true" : "false"
        );
      if (filterSettings.config.history)
        params.append(
          "use_history",
          filterSettings.config.history ? "true" : "false"
        );
      if (filterSettings.config.normalized)
        params.append(
          "normalize",
          filterSettings.config.normalized ? "true" : "false"
        );
      if (filterSettings.config.messageCount)
        params.append("history_length", filterSettings.config.messageCount);
      
      if (filterSettings.config.history) {
        const messageWeights = filterSettings.config.messageWeights || [0.2, 0.3, 0.5];
        const formattedWeights = messageWeights.map(weight => parseFloat(weight.toFixed(1)));
        params.append("message_weights", JSON.stringify(formattedWeights));
      }
    }
    return params;
  };

  const analyzeComparisonNetwork = async (
    index,
    globalFilterParams,
    customFilters = null,
    isWikipediaData = false,
    wikiSection = null
  ) => {
    const comparisonFile = comparisonData[index];
    if (
      !comparisonFile ||
      (!comparisonFile.filename && !comparisonFile.isWikipediaData)
    ) {
      return {
        success: false,
        message: "Please select a comparison file first.",
      };
    }
    let params = new URLSearchParams(globalFilterParams?.toString() || "");

    if (customFilters) {
      updateComparisonFilterSettings(index, customFilters);
      const customParams = buildFilterParamsFromSettings(customFilters);
      for (const [key, value] of customParams.entries()) {
        params.set(key, value);
      }
    } else if (comparisonFilterSettings[index]) {
      const savedParams = buildFilterParamsFromSettings(
        comparisonFilterSettings[index]
      );
      for (const [key, value] of savedParams.entries()) {
        params.set(key, value);
      }
    }

    let url;
    if (isWikipediaData || comparisonFile.isWikipediaData) {
      url = `${BASE_URL}/analyze/wikipedia/${wikiSection}?${params.toString()}`;
    } else {
      url = `${BASE_URL}/analyze/network/${comparisonFile.filename
        }?${params.toString()}`;
    }

    try {
      const response = await fetch(url);
      const data = await response.json();

      if (data.nodes && data.links) {
        const updatedComparisonData = [...comparisonNetworkData];
        updatedComparisonData[index] = data;
        setComparisonNetworkData(updatedComparisonData);

        const updatedData = [...comparisonData];
        updatedData[index] = {
          ...updatedData[index],
          isAnalyzed: true,
        };
        setComparisonData(updatedData);

        if (!activeComparisonIndices.includes(index)) {
          setActiveComparisonIndices((prev) => [...prev, index]);
        }

        return {
          success: true,
          message: `Comparison analysis ${index + 1} completed successfully!`,
        };
      } else {
        return {
          success: false,
          message: `No valid data returned for comparison ${index + 1}.`,
        };
      }
    } catch (error) {
      console.error(`Error during comparison analysis ${index}:`, error);
      return {
        success: false,
        message: `Error analyzing comparison file ${index + 1}.`,
      };
    }
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

  const ensureMetricsData = (graphData, metrics) => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      return;
    }

    if (
      metrics.includes("Degree Centrality") &&
      !graphData.nodes.some((node) => node.hasOwnProperty("degree"))
    ) {
      const degreeMap = {};
      graphData.nodes.forEach((node) => (degreeMap[node.id] = 0));

      graphData.links.forEach((link) => {
        const sourceId =
          typeof link.source === "object" ? link.source.id : link.source;
        const targetId =
          typeof link.target === "object" ? link.target.id : link.target;

        degreeMap[sourceId] = (degreeMap[sourceId] || 0) + 1;
        degreeMap[targetId] = (degreeMap[targetId] || 0) + 1;
      });

      graphData.nodes.forEach((node) => {
        node.degree = degreeMap[node.id] || 0;
      });
    }
  };

  const applyComparisonFilters = async (filters) => {
    if (
      !originalNetworkData ||
      activeComparisonIndices.length === 0 ||
      !uploadedFile
    ) {
      return { success: false, message: "No data available for comparison." };
    }

    setComparisonFilter(filters.comparisonFilter);
    setMinComparisonWeight(filters.minComparisonWeight);
    setComparisonMetrics(filters.comparisonMetrics);
    setHighlightCommonNodes(filters.highlightCommonNodes);

    const params = new URLSearchParams();
    if (filters.comparisonFilter)
      params.append("node_filter", filters.comparisonFilter);
    if (filters.minComparisonWeight)
      params.append("min_weight", filters.minComparisonWeight);
    params.append("highlight_common", filters.highlightCommonNodes.toString());
    params.append("original_filename", uploadedFile);

    if (filters.networkFilterParams) {
      for (const [key, value] of filters.networkFilterParams.entries()) {
        if (!params.has(key)) {
          params.append(key, value);
        }
      }
    }

    if (filters.comparisonMetrics && filters.comparisonMetrics.length > 0) {
      params.append("metrics", filters.comparisonMetrics.join(","));
    }

    try {
      const comparisonPromises = activeComparisonIndices.map((index) => {
        const comparisonParams = new URLSearchParams(params);
        const comparisonFile = comparisonData[index];

        if (comparisonFile.isWikipediaData) {
          const filename = comparisonFile.isOriginalFile
            ? "wikipedia_data"
            : "comparison_wikipedia_data";
          comparisonParams.append("comparison_filename", filename);
        } else {
          comparisonParams.append(
            "comparison_filename",
            comparisonFile.filename
          );
        }

        if (comparisonFilterSettings[index]) {
          const fileSpecificParams = buildFilterParamsFromSettings(
            comparisonFilterSettings[index]
          );
          for (const [key, value] of fileSpecificParams.entries()) {
            comparisonParams.set(key, value);
          }
        }

        return fetch(
          `${BASE_URL}/analyze/compare-networks?${comparisonParams.toString()}`
        ).then((response) => response.json());
      });

      const results = await Promise.all(comparisonPromises);

      if (results.some((data) => data.error)) {
        const errors = results
          .filter((data) => data.error)
          .map((data) => data.error)
          .join(", ");
        return { success: false, message: `Error in comparisons: ${errors}` };
      }

      const processedOriginal = standardizeGraphData(results[0].original);

      if (filters.comparisonMetrics && filters.comparisonMetrics.length > 0) {
        ensureMetricsData(processedOriginal, filters.comparisonMetrics);
      }

      setFilteredOriginalData(processedOriginal);

      const processedComparisons = {};
      activeComparisonIndices.forEach((index, arrayIndex) => {
        const processedComparison = standardizeGraphData(
          results[arrayIndex].comparison
        );

        if (filters.comparisonMetrics && filters.comparisonMetrics.length > 0) {
          ensureMetricsData(processedComparison, filters.comparisonMetrics);
        }

        processedComparisons[index] = processedComparison;
      });

      setFilteredComparisonData(processedComparisons);

      return {
        success: true,
        message: `${results.length} network comparisons completed successfully!`,
      };
    } catch (error) {
      console.error("Error during network comparisons:", error);
      return {
        success: false,
        message: "An error occurred during network comparisons",
      };
    }
  };

  const resetComparisonFilters = () => {
    setFilteredOriginalData(null);
    setFilteredComparisonData({});
    setComparisonFilter("");
    setMinComparisonWeight(1);
    setComparisonMetrics([]);
    setHighlightCommonNodes(false);
  };

  const resetItemFilters = (index) => {
    if (index !== undefined) {
      updateComparisonFilterSettings(index, getDefaultFilterSettings());
    } else {
      setComparisonFilterSettings((prevSettings) =>
        prevSettings.map(() => getDefaultFilterSettings())
      );
    }
  };

  const calculateComparisonStats = (originalData, comparisonNetworkItem) => {
    if (!originalData || !comparisonNetworkItem) {
      return null;
    }

    const originalNodeCount = originalData.nodes.length;
    const comparisonNodeCount = comparisonNetworkItem.nodes.length;
    const originalLinkCount = originalData.links.length;
    const comparisonLinkCount = comparisonNetworkItem.links.length;

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
      comparisonNetworkItem.nodes.map((node) => node.id)
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

  const addWikipediaComparison = (index, wikiData, isOriginal = false) => {
    const updatedData = [...comparisonData];
    updatedData[index] = {
      id: index,
      filename: isOriginal
        ? "wikipedia_data"
        : `comparison_wikipedia_data_${index}`,
      name: isOriginal
        ? "Original Wikipedia Discussion"
        : "Wikipedia Discussion",
      isWikipediaData: true,
      isOriginalFile: isOriginal,
      isAnalyzed: false,
      wikiContent: wikiData.content,
      selectedSection: wikiData.selectedSection,
    };
    setComparisonData(updatedData);

    if (!comparisonFilterSettings[index]) {
      updateComparisonFilterSettings(index, getDefaultFilterSettings());
    }
  };

  return {
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
    comparisonFilterSettings,
    addComparison,
    handleComparisonFileChange,
    toggleComparisonActive,
    analyzeComparisonNetwork,
    applyComparisonFilters,
    resetComparisonFilters,
    resetItemFilters,
    updateComparisonFilterSettings,
    calculateComparisonStats,
    setComparisonData,
    setComparisonFiles,
    addWikipediaComparison,
  };
};

export default useComparison;
