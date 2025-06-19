import React, { useState, useEffect } from "react";
import {
  Card,
  Alert,
  Button,
  Row,
  Col,
  OverlayTrigger,
  Tooltip,
} from "react-bootstrap";
import {
  FileBarGraph,
  PlusCircle,
  GraphUp,
  ArrowsFullscreen,
  ArrowsCollapse,
  InfoCircle,
} from "react-bootstrap-icons";
import { toast } from "sonner";
import { useDispatch } from "react-redux";
import ComparisonItem from "../../components/comparison/ComparisonItem";
import ComparisonMetrics from "../../components/comparison/ComparisonMetrics";
import "../../styles/ComparativeAnalysis.css";

const ComparativeAnalysis = ({
  originalNetworkData,
  comparison,
  filters,
  uploadedFileName,
  platform,
  formData,
  setFormData,
  wikiContent,
  setWikiContent,
  selectedSection,
  setSelectedSection,
  handleFetchWikipedia,
}) => {
  const dispatch = useDispatch();
  const [message, setMessage] = useState("");
  const [expandedView, setExpandedView] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeFilters, setActiveFilters] = useState({});
  const [localFilterSettings, setLocalFilterSettings] = useState([]);
  const [comparisonMetrics, setComparisonMetrics] = useState([]);

  const {
    comparisonCount,
    comparisonFiles,
    comparisonData,
    comparisonNetworkData,
    activeComparisonIndices,
    filteredOriginalData,
    filteredComparisonData,
    addComparison,
    handleComparisonFileChange,
    toggleComparisonActive,
    analyzeComparisonNetwork,
    applyComparisonFilters,
    resetComparisonFilters,
    updateComparisonFilterSettings,
    setComparisonData,
    setComparisonFiles,
    comparisonFilterSettings,
  } = comparison;



  const graphMetrics = [
    "Degree Centrality",
    "Betweenness Centrality",
    "Closeness Centrality",
    "Eigenvector Centrality",
    "PageRank Centrality",
  ];

  useEffect(() => {
    if (comparisonCount > localFilterSettings.length) {
      const newFilterSettings = [...localFilterSettings];
      for (let i = localFilterSettings.length; i < comparisonCount; i++) {
        newFilterSettings.push({
          timeFrame: {
            startDate: "",
            endDate: "",
            startTime: "",
            endTime: "",
          },
          limit: {
            enabled: true,
            count: 50,
            fromEnd: false,
            type: "messages",
          },
          config: {
            directed: false,
            anonymize: false,
            history: false,
            messageCount: 3,
            normalized: false,
            messageWeights: [0.5, 0.3, 0.2],
          },
        });
      }
      setLocalFilterSettings(newFilterSettings);
    }
  }, [comparisonCount]);

  const handleFilterChange = (index, newFilters) => {
    const updatedFilterSettings = [...localFilterSettings];
    updatedFilterSettings[index] = newFilters;
    setLocalFilterSettings(() => updatedFilterSettings);
    checkActiveFilters(index, newFilters);

    if (newFilters.filterUpdated) {
      const updatedData = [...comparisonData];
      updatedData[index] = {
        ...updatedData[index],
        ...newFilters,
        filterUpdated: undefined,
      };
      setComparisonData(() => updatedData);
    }
  };

  const checkActiveFilters = (index, filters) => {
    const hasTimeFrame =
      filters.timeFrame?.startDate ||
      filters.timeFrame?.endDate ||
      filters.timeFrame?.startTime ||
      filters.timeFrame?.endTime;
    const hasLimit = filters.limit?.enabled;
    const isActive = hasTimeFrame || hasLimit;
    setActiveFilters((prev) => ({
      ...prev,
      [index]: isActive,
    }));
  };

  const handleAnalyzeNetwork = (
    index,
    itemFilters,
    isWikipediaData = false
  ) => {
    
    const sumWeight = itemFilters?.config?.messageWeights?.reduce(
      (acc, weight) => acc + weight,
      0
    );
    if (sumWeight !== 1) {
      toast.error(
        "Message weights must sum to 1. Please adjust the weights."
      );
      return;
    }

    setIsAnalyzing(true);


    const actualIsWikipediaData =
      isWikipediaData ||
      comparisonData[index]?.isWikipediaData ||
      platform === "wikipedia";

    return analyzeComparisonNetwork(
      index,
      null,
      itemFilters,
      actualIsWikipediaData,
      selectedSection?.title
    )
      .then((result) => {
        if (result && result.success) {
          setMessage(result.message || "Network analyzed successfully");
          toast.success(result.message || "Network analyzed successfully");
          updateComparisonFilterSettings(index, itemFilters);
          return applyComparisonFilters({
            comparisonMetrics,
          });
        } else {
          toast.error(result.message || "Analysis failed");
        }
        return result;
      })
      .finally(() => {
        setIsAnalyzing(false);
      });
  };

  const handleAddComparison = () => {
    addComparison();
    toast.success("New comparison item added");
  };

  const toggleComparisonMetric = (metric) => {
    setComparisonMetrics((prev) =>
      prev.includes(metric)
        ? prev.filter((m) => m !== metric)
        : [...prev, metric]
    );
  };

  const handleResetVisualizationFilters = () => {
    setComparisonMetrics([]);
    if (resetComparisonFilters) {
      resetComparisonFilters();
      toast.success("Visualization filters reset");
    }
  };

  const getActiveFilterCount = (index) => {
    const filters = localFilterSettings[index];
    if (!filters) return 0;
    let count = 0;
    if (filters.timeFrame?.startDate) count++;
    if (filters.timeFrame?.endDate) count++;
    if (filters.timeFrame?.startTime) count++;
    if (filters.timeFrame?.endTime) count++;
    if (filters.limit?.enabled) count++;
    return count;
  };

  const toggleExpandedView = () => {
    setExpandedView(!expandedView);
  };

  const useOriginalFile = (index, wikiData = null) => {
    if (platform === "wikipedia") {
      let updatedData;

      if (wikiData) {
        updatedData = {
          id: index,
          name: wikiData.selectedSection
            ? `Wikipedia: ${wikiData.selectedSection.title}`
            : "Wikipedia Discussion (Please select a section)",
          filename: wikiData.filename || `comparison_wikipedia_data_${index}`,
          isWikipediaData: true,
          isOriginalFile: false,
          isAnalyzed: false,
          wikiContent: wikiData.content,
          selectedSection: wikiData.selectedSection,
          rawData: wikiData.rawData, 
        };

        comparisonFiles[index] =
          wikiData.filename || `comparison_wikipedia_data_${index}`;
      } else {
        updatedData = {
          id: index,
          name: selectedSection
            ? `Original Wikipedia: ${selectedSection.title}`
            : "Original Wikipedia Discussion",
          filename: "wikipedia_data",
          isWikipediaData: true,
          isOriginalFile: true,
          isAnalyzed: false,
          wikiContent: wikiContent,
          selectedSection: selectedSection,
        };

        comparisonFiles[index] = "wikipedia_data";
      }

      comparisonData[index] = updatedData;
    } else {
      const updatedData = {
        name: uploadedFileName,
        filename: uploadedFileName,
        isOriginalFile: true,
        isAnalyzed: false,
      };

      comparisonData[index] = updatedData;
      comparisonFiles[index] = uploadedFileName;
    }

    updateComparisonFilterSettings(index, {
      timeFrame: {
        startDate: "",
        endDate: "",
        startTime: "",
        endTime: "",
      },
      limit: {
        enabled: true,
        count: 50,
        fromEnd: false,
        type: "messages",
      },
      config: {
        directed: false,
        anonymize: false,
        history: false,
        messageCount: 3,
        normalized: false,
        messageWeights: [0.5, 0.3, 0.2],
      },
    });

    setComparisonData([...comparisonData]);
    setComparisonFiles([...comparisonFiles]);
  };

  const handleWikipediaFetch = async (url) => {
    try {
      const data = await handleFetchWikipedia(url);
      return data;
    } catch (error) {
      console.error("Error fetching Wikipedia data:", error);
      throw error;
    }
  };

  return (
    <Card className="research-card">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-start mb-4">
          <div>
            <h3 className="step-title">Comparative Analysis</h3>
            <p className="text-muted">
              Compare your current network with other networks to identify
              patterns, changes, and differences over time.
            </p>
          </div>
        </div>

        <Alert variant="info" className="d-flex align-items-start mb-4">
          <InfoCircle size={20} className="me-2 mt-1" />
          <div>
            <strong>Pro tip:</strong>
            {platform === "wikipedia"
              ? " For Wikipedia analysis, you can compare different discussion sections from the same page or different Wikipedia pages to see how discussions vary."
              : " For time-series analysis, reuse the original dataset with different time filters to compare how the network evolves over time."}
          </div>
        </Alert>

        <div className={`comparison-section ${expandedView ? "expanded" : ""}`}>
          <div className="section-header d-flex justify-content-between align-items-center mb-3">
            <h4 className="mb-0">
              <FileBarGraph className="me-2" /> Comparison Files
            </h4>
            <Button
              variant="primary"
              className="btn-with-icon"
              onClick={handleAddComparison}
            >
              <PlusCircle className="me-1" /> Add Comparison
            </Button>
          </div>

          {comparisonCount === 0 ? (
            <Alert variant="light" className="text-center py-4">
              <div className="mb-3">
                <FileBarGraph size={32} className="text-secondary" />
              </div>
              <h5>No comparison files added yet</h5>
              <p className="text-muted mb-3">
                {platform === "wikipedia"
                  ? "Add a comparison to analyze different Wikipedia discussions or sections"
                  : "Add a comparison file to start analyzing differences between networks"}
              </p>
              <Button variant="primary" onClick={handleAddComparison}>
                <PlusCircle className="me-2" /> Add Comparison File
              </Button>
            </Alert>
          ) : (
            <div className="comparison-files-container">
              {[...Array(comparisonCount)].map((_, index) => (
                <ComparisonItem
                  key={index}
                  index={index}
                  comparisonData={comparisonData[index]}
                  comparisonFile={comparisonFiles[index]}
                  filterSettings={localFilterSettings[index]}
                  onFileUpload={(e) => handleComparisonFileChange(e, index,platform)}
                  onAnalyzeNetwork={(index, filters) =>
                    handleAnalyzeNetwork(index, filters)
                  }
                  onFilterChange={handleFilterChange}
                  isActive={activeComparisonIndices.includes(index)}
                  onToggleActive={() => toggleComparisonActive(index)}
                  isAnalyzing={isAnalyzing}
                  activeFilterCount={getActiveFilterCount(index)}
                  hasActiveFilters={activeFilters[index]}
                  originalFileName={uploadedFileName}
                  useOriginalFile={useOriginalFile}
                  platform={platform}
                  formData={formData}
                  setFormData={setFormData}
                  wikiContent={wikiContent}
                  setWikiContent={setWikiContent}
                  selectedSection={selectedSection}
                  setSelectedSection={setSelectedSection}
                  handleFetchWikipedia={handleWikipediaFetch}
                />
              ))}
            </div>
          )}
        </div>

        {activeComparisonIndices.length > 0 && originalNetworkData ? (
          <div className="mt-4">
            <div className="section-header mb-3">
              <h4 className="mb-0">
                <GraphUp className="me-2" /> Network Comparison Visualization
              </h4>
            </div>
            <ComparisonMetrics
              originalNetworkData={originalNetworkData}
              comparisonNetworkData={comparisonNetworkData}
              comparisonData={comparisonData}
              activeComparisonIndices={activeComparisonIndices}
              onToggleComparison={toggleComparisonActive}
              filteredOriginalData={filteredOriginalData}
              filteredComparisonData={filteredComparisonData}
              onResetComparisonFilters={resetComparisonFilters}
              comparisonFilterSettings={comparisonFilterSettings}
              originalFilters={filters}
              formData={formData}
            />
          </div>
        ) : (
          <Alert variant="light" className="text-center py-4 mt-4">
            <div className="mb-3">
              <GraphUp size={40} className="text-secondary" />
            </div>
            <h5>No active comparisons to visualize</h5>
            <p className="text-muted mb-3">
              {comparisonCount === 0
                ? platform === "wikipedia"
                  ? "Add and analyze Wikipedia discussions first"
                  : "Add and analyze comparison files first"
                : "Analyze files and mark them as active for comparison"}
            </p>
          </Alert>
        )}

        {message && <div className="alert alert-info mt-3">{message}</div>}
      </Card.Body>
    </Card>
  );
};

export default ComparativeAnalysis;
