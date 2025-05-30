import React, { useState, useRef, useEffect } from "react";
import { Container, Row, Col, Button, Modal } from "react-bootstrap";
import { ChevronLeft, ChevronRight } from "react-bootstrap-icons";
import { toast } from "sonner";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";
import useFilters from "../hooks/useFilters";
import useComparison from "../hooks/useComparison";
import { clearImages } from "../redux/images/imagesSlice";

import ResearchSetup from "./Steps/ResearchSetup";
import Discussion from "./Steps/Discussion";
import DataConfiguration from "./Steps/DataConfiguration";
import TimeFrame from "./Steps/TimeFrame";
import MessageContent from "./Steps/MessageContent";
import UserFilters from "./Steps/UserFilters";
import NetworkVisualization from "./Steps/NetworkVisualization";
import ComparativeAnalysis from "./Steps/ComparativeAnalysis";
import ResearchReport from "./Steps/ResearchReport";

import {
  uploadFile,
  analyzeNetwork,
  detectCommunities,
  fetchWikipediaData,
  // analyzeWikipediaNetwork,
  detectWikipediaCommunities,
} from "../components/utils/ApiService";
import { saveToDB } from "../components/utils/ApiService";

import "../styles/ResearchWizard.css";

const ResearchWizard = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const selectedPlatform = location.state?.platform || "whatsapp";
  const dispatch = useDispatch();
  const fileInputRef = useRef(null);
  const { currentUser, token } = useSelector((state) => state.user) || { id: 1 };
  const [currentStep, setCurrentStep] = useState(1);
  const [wikiContent, setWikiContent] = useState(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const hasShownToastRef = useRef(false);

  const ALL_STEPS = {
    SETUP: "Setup",
    WIKIPEDIA: "Discussion",
    DATA_CONFIG: "DataConfiguration",
    TIME_FRAME: "TimeFrame",
    MESSAGE_CONTENT: "MessageContent",
    USER_FILTERS: "UserFilters",
    NETWORK_VISUALIZATION: "NetworkVisualization",
    COMPARATIVE_ANALYSIS: "ComparativeAnalysis",
    RESEARCH_REPORT: "ResearchReport",
  };

  const STEP_LABELS = {
    [ALL_STEPS.SETUP]: "Setup",
    [ALL_STEPS.WIKIPEDIA]: "Discussion",
    [ALL_STEPS.DATA_CONFIG]: "Config",
    [ALL_STEPS.TIME_FRAME]: "Time",
    [ALL_STEPS.MESSAGE_CONTENT]: "Content",
    [ALL_STEPS.USER_FILTERS]: "Users",
    [ALL_STEPS.NETWORK_VISUALIZATION]: "Network",
    [ALL_STEPS.COMPARATIVE_ANALYSIS]: "Compare",
    [ALL_STEPS.RESEARCH_REPORT]: "Report",
  };

  const STEP_TITLES = {
    [ALL_STEPS.SETUP]: "New Research - Setup",
    [ALL_STEPS.WIKIPEDIA]: "New Research - Discussion",
    [ALL_STEPS.DATA_CONFIG]: "New Research - Data Configuration",
    [ALL_STEPS.TIME_FRAME]: "New Research - Time Frame",
    [ALL_STEPS.MESSAGE_CONTENT]: "New Research - Message Content",
    [ALL_STEPS.USER_FILTERS]: "New Research - User Filters",
    [ALL_STEPS.NETWORK_VISUALIZATION]: "New Research - Network Visualization",
    [ALL_STEPS.COMPARATIVE_ANALYSIS]: "New Research - Comparative Analysis",
    [ALL_STEPS.RESEARCH_REPORT]: "New Research - Research Report",
  };

  const [networkData, setNetworkData] = useState(null);
  const [originalNetworkData, setOriginalNetworkData] = useState(null);
  const [shouldFetchCommunities, setShouldFetchCommunities] = useState(false);
  const [communities, setCommunities] = useState([]);
  const [communityMap, setCommunityMap] = useState({});
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [message, setMessage] = useState("");
  const [shouldShowUserFilters, setShouldShowUserFilters] = useState(true);
  const [shouldShowMessageContent, setShouldShowMessageContent] =
    useState(true);
  const [lastAnalysisParams, setLastAnalysisParams] = useState(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showLoginInvite, setShowLoginInvite] = useState(false);

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    file: null,
    fileName: "",
    uploadedFileName: "",
    platform: selectedPlatform,
    wikipediaUrl: "",
    isAnonymized: false,
    includeMessageContent: true,
    isDirectedGraph: false,
    useHistoryAlgorithm: false,
    isNormalized: false,
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
    messageCriteria: {
      minLength: 1,
      maxLength: "",
      keywords: "",
      contentFilter: "",
    },
    userFilters: {
      minMessages: 1,
      maxMessages: "",
      activeUsers: 0,
      usernameFilter: "",
    },
    Network: {
      degreeCentrality: true,
      betweennessCentrality: true,
      closenessCentrality: true,
      eigenvectorCentrality: true,
      pagerankCentrality: true,
      communityDetection: true,
    },
    comparisonEnabled: false,
    comparisonFile: null,
    comparisonFileName: "",
    comparisonUploadedFileName: "",
  });

  const filters = useFilters(formData);

  const comparison = useComparison(
    originalNetworkData,
    formData.uploadedFileName
  );

  const getVisibleSteps = () => {
    const steps = [];
    steps.push(ALL_STEPS.SETUP);

    if (selectedPlatform === "wikipedia") {
      steps.push(ALL_STEPS.WIKIPEDIA);
    }

    steps.push(ALL_STEPS.DATA_CONFIG);
    steps.push(ALL_STEPS.TIME_FRAME);
    if (shouldShowMessageContent) {
      steps.push(ALL_STEPS.MESSAGE_CONTENT);
    }

    if (shouldShowUserFilters) {
      steps.push(ALL_STEPS.USER_FILTERS);
    }
    steps.push(ALL_STEPS.NETWORK_VISUALIZATION);
    steps.push(ALL_STEPS.COMPARATIVE_ANALYSIS);
    steps.push(ALL_STEPS.RESEARCH_REPORT);

    return steps;
  };

  const getCurrentStepContent = () => {
    const visibleSteps = getVisibleSteps();
    return visibleSteps[currentStep - 1] || ALL_STEPS.SETUP;
  };

  const getVisibleTotalSteps = () => {
    return getVisibleSteps().length;
  };

  useEffect(() => {
    const currentStepContent = getCurrentStepContent();
    document.title = STEP_TITLES[currentStepContent] || "New Research";
  }, [currentStep]);

  useEffect(() => {
    if (networkData && shouldFetchCommunities) {
      fetchCommunityData();
      setShouldFetchCommunities(false);
    }
  }, [networkData, shouldFetchCommunities]);

  useEffect(() => {
    if (formData.isDirectedGraph && formData.useHistoryAlgorithm) {
      setShouldShowUserFilters(false);
      setShouldShowMessageContent(false);
    } else {
      setShouldShowUserFilters(true);
      setShouldShowMessageContent(formData.includeMessageContent);
    }
  }, [
    formData.isDirectedGraph,
    formData.useHistoryAlgorithm,
    formData.includeMessageContent,
  ]);

  useEffect(() => {
    const currentStepContent = getCurrentStepContent();
    const isNetworkVisualizationStep =
      currentStepContent === ALL_STEPS.NETWORK_VISUALIZATION;

    if (isNetworkVisualizationStep && formData.uploadedFileName) {
      handleReanalysis();
    }
  }, [
    currentStep,
    formData.uploadedFileName,
    shouldShowUserFilters,
    formData.includeMessageContent,
    formData.limit.enabled,
    formData.limit.count,
    formData.limit.fromEnd,
    formData.timeFrame.startDate,
    formData.timeFrame.endDate,
    formData.timeFrame.startTime,
    formData.timeFrame.endTime,
    formData.messageCriteria.minLength,
    formData.messageCriteria.maxLength,
    formData.messageCriteria.keywords,
    formData.messageCriteria.contentFilter,
    formData.userFilters.minMessages,
    formData.userFilters.maxMessages,
    formData.userFilters.activeUsers,
    formData.userFilters.usernameFilter,
    formData.isDirectedGraph,
    formData.useHistoryAlgorithm,
    formData.isNormalized,
  ]);

  useEffect(() => {
    const currentStepContent = getCurrentStepContent();

    const visibleSteps = getVisibleSteps();
    const networkStepIndex = visibleSteps.indexOf(
      ALL_STEPS.NETWORK_VISUALIZATION
    );
    const isAfterNetworkStep = currentStep > networkStepIndex + 1;

    if (
      (currentStepContent === ALL_STEPS.NETWORK_VISUALIZATION ||
        isAfterNetworkStep) &&
      formData.uploadedFileName &&
      shouldReanalyze()
    ) {
      const timeoutId = setTimeout(() => {
        handleReanalysis();
      }, 500);

      return () => clearTimeout(timeoutId);
    }
  }, [
    formData.limit.enabled,
    formData.limit.count,
    formData.limit.fromEnd,
    formData.timeFrame.startDate,
    formData.timeFrame.endDate,
    formData.timeFrame.startTime,
    formData.timeFrame.endTime,
    formData.messageCriteria.minLength,
    formData.messageCriteria.maxLength,
    formData.messageCriteria.keywords,
    formData.messageCriteria.contentFilter,
    formData.userFilters.minMessages,
    formData.userFilters.maxMessages,
    formData.userFilters.activeUsers,
    formData.userFilters.usernameFilter,
    formData.isDirectedGraph,
    formData.useHistoryAlgorithm,
    formData.isNormalized,
  ]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFormData({
      ...formData,
      file: selectedFile,
      fileName: selectedFile.name,
    });
    handleFileUpload(selectedFile);
  };

  const handleFileUpload = (file) => {
    if (!file) {
      toast.error("Please select a file to upload.");
      return;
    }
    toast.promise(uploadFile(file), {
      loading: "Uploading file...",
      success: (data) => {
        setFormData((prev) => ({
          ...prev,
          uploadedFileName: data.filename,
        }));
        return "File uploaded successfully!";
      },
      error: (error) => {
        return error?.message || "Error uploading file.";
      },
    });
  };

  const handleFetchWikipedia = () => {
    if (!formData.wikipediaUrl?.trim()) {
      toast.error("Please enter a valid Wikipedia URL.");
      return;
    }

    toast.promise(
      fetchWikipediaData(formData.wikipediaUrl).then(async (data) => {
        if (data.nodes && data.links) {
          setNetworkData(data);
          setOriginalNetworkData(data);
          setWikiContent(data.content);

          setFormData((prev) => ({
            ...prev,
            uploadedFileName: "wikipedia_data.txt",
          }));

          await fetch(
            `${import.meta.env.VITE_API_URL}/convert-wikipedia-to-txt`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                filename: "wikipedia_data",
                section_title: "Top",
              }),
            }
          );

          return data;
        } else {
          throw new Error("No valid discussion data found.");
        }
      }),
      {
        loading: "Fetching discussion...",
        success: "Wikipedia discussion loaded and converted!",
        error: (err) => err.message || "Error fetching Wikipedia data.",
      }
    );
  };

  const handleNetworkAnalysis = async () => {
    if (!formData.uploadedFileName) {
      toast.error("No file selected for analysis.");
      return;
    }
  
    const params = filters.buildNetworkFilterParams();
    const platformParam = `platform=${formData.platform}`;
    const finalParams = `${params}&${platformParam}`;
    setLastAnalysisParams(finalParams);
  
    try {
      if (formData.platform === "wikipedia") {
        await fetch(`${import.meta.env.VITE_API_URL}/convert-wikipedia-to-txt`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: formData.uploadedFileName,
            section_title: selectedSection || "Top",
          }),
        });
  
        // toast.promise(analyzeWikipediaNetwork("wikipedia_data", finalParams), {
          toast.promise(analyzeNetwork("wikipedia_data", finalParams), {

          loading: "Analyzing Wikipedia discussion...",
          success: async (data) => {
            if (data.nodes && data.links) {
              dispatch(clearImages());
  
              const communityData = await detectWikipediaCommunities("wikipedia_data", finalParams);
              const nodeCommunities = communityData.node_communities || {};
              const updatedNodes = data.nodes.map((node) => {
                const community = nodeCommunities[node.id?.toString().trim()];
                return community !== undefined ? { ...node, community } : node;
              });
  
              data.nodes = updatedNodes;
              setNetworkData(data);
              setOriginalNetworkData(data);
              setCommunities(communityData.communities || []);
              setCommunityMap(nodeCommunities);
  
              return "Wikipedia analysis completed successfully!";
            } else {
              return "No data returned from Wikipedia analysis.";
            }
          },
          error: (error) => error?.message || "Error analyzing Wikipedia discussion.",
        });
  
      } else if (formData.platform === "whatsapp") {
        if (hasShownToastRef.current) return;
        hasShownToastRef.current = true;
  
        toast.promise(analyzeNetwork(formData.uploadedFileName, finalParams), {
          loading: "Analyzing WhatsApp network...",
          success: async (data) => {
            if (data.nodes && data.links) {
              dispatch(clearImages());
              setNetworkData(data);
              setOriginalNetworkData(data);
  
              const communityData = await detectCommunities(formData.uploadedFileName, finalParams);
              setCommunities(communityData.communities || []);
              setCommunityMap(communityData.node_communities || {});
              setShouldFetchCommunities(true);
  
              return "WhatsApp analysis completed successfully!";
            } else {
              return "No data returned from server.";
            }
          },
          error: (error) => error?.message || "Error analyzing network.",
        });
  
      } else {
        toast.error("Unsupported platform selected.");
      }
  
    } catch (error) {
      console.error("Error during network analysis:", error);
      toast.error("Failed to analyze network.");
    }
  };
  
  const fetchCommunityData = () => {
    if (!formData.uploadedFileName || !networkData) return;
  
    const params = filters.buildNetworkFilterParams();
    const platformParam = `platform=${formData.platform}`;
    const finalParams = `${params}&${platformParam}`;
  
    detectCommunities(formData.uploadedFileName, finalParams)
      .then((data) => {
        if (data.communities && data.nodes) {
          setCommunities(data.communities);
          const newCommunityMap = {};
          data.nodes.forEach((node) => {
            if (node.community !== undefined) {
              newCommunityMap[node.id.toString().trim()] = node.community;
            }
          });
          setCommunityMap(newCommunityMap);
  
          const updatedNodes = networkData.nodes.map((node) => {
            const normalizedId = node.id.toString().trim();
            const community = newCommunityMap[normalizedId];
            return community !== undefined ? { ...node, community } : node;
          });
  
          setNetworkData({ nodes: updatedNodes, links: networkData.links });
          setOriginalNetworkData({ nodes: updatedNodes, links: networkData.links });
        }
      })
      .catch((error) => {
        toast.error(error.message || "Error detecting communities.");
      });
  };
  
  const handleReanalysis = () => {
    if (shouldReanalyze()) {
      setNetworkData(null);
      setOriginalNetworkData(null);
      setCommunities([]);
      setCommunityMap({});
      hasShownToastRef.current = false;
      handleNetworkAnalysis();
    }
  };

  const shouldReanalyze = () => {
    if (!formData.uploadedFileName || !networkData) return false;

    const currentParams = filters.buildNetworkFilterParams().toString();
    return !lastAnalysisParams || currentParams !== lastAnalysisParams;
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;

    if (name.includes(".")) {
      const [parent, child] = name.split(".");
      let newValue;
      if (type === "checkbox") {
        newValue = checked;
      } else if (type === "radio") {
        newValue = value === "true" ? true : value === "false" ? false : value;
      } else if (type === "number") {
        newValue = value === "" ? "" : Number(value);
      } else {
        newValue = value;
      }
      setFormData({
        ...formData,
        [parent]: {
          ...formData[parent],
          [child]: newValue,
        },
      });
    } else {
      setFormData({
        ...formData,
        [name]: type === "checkbox" ? checked : value,
      });
    }

    if (name === "isDirectedGraph" || name === "useHistoryAlgorithm") {
      const currentStepContent = getCurrentStepContent();

      if (
        (checked &&
          name === "useHistoryAlgorithm" &&
          formData.isDirectedGraph) ||
        (checked && name === "isDirectedGraph" && formData.useHistoryAlgorithm)
      ) {
        if (
          currentStepContent === ALL_STEPS.MESSAGE_CONTENT ||
          currentStepContent === ALL_STEPS.USER_FILTERS
        ) {
          const visibleSteps = getVisibleSteps();
          const networkVisIndex = visibleSteps.indexOf(
            ALL_STEPS.NETWORK_VISUALIZATION
          );
          if (networkVisIndex !== -1) {
            setCurrentStep(networkVisIndex + 1);
          }
        }
      }
    }
  };

  const handleSaveResearch = async () => {
    const params = filters.buildNetworkFilterParams();
    const id = currentUser?.id;
    if (!formData.name || !formData.uploadedFileName || !params || !id) {
      toast.error("Please fill in all required fields.");
      return null;
    }

    const result = await toast.promise(
      saveToDB(
        id,
        token,
        formData.name,
        formData.description,
        formData.uploadedFileName,
        params,
        selectedMetric,
        {
          hasComparison: comparison.comparisonNetworkData?.length > 0,
          data: comparison.comparisonNetworkData,
        },
        formData.platform,
        communities
      ),
      {
        loading: "Saving research...",
        success: "Research saved successfully!",
        error: (error) => error?.detail || "Error saving research.",
      }
    );

    return result;
  };

  const goToNextStep = () => {
    if (currentStep === 1) {
      if (!formData.name.trim()) {
        toast.error("Please enter a research name.");
        return;
      }
      if (!formData.uploadedFileName) {
        toast.error("Please upload a file to continue.");
        return;
      }
    }

    const visibleSteps = getVisibleSteps();
    const nextStep = currentStep + 1;

    if (
      nextStep <= visibleSteps.length &&
      visibleSteps[nextStep - 1] === ALL_STEPS.NETWORK_VISUALIZATION
    ) {
      if (shouldReanalyze()) {
        handleReanalysis();
      }
    }

    if (currentStep < visibleSteps.length) {
      setCurrentStep(nextStep);
    }
  };

  const goToPreviousStep = () => {
    if (currentStep > 1) {
      const currentStepContent = getCurrentStepContent();
      if (currentStepContent === ALL_STEPS.NETWORK_VISUALIZATION) {
        setLastAnalysisParams(null);
      }

      setCurrentStep(currentStep - 1);
    }
  };

  const renderCurrentStep = () => {
    const currentStepContent = getCurrentStepContent();

    switch (currentStepContent) {
      case ALL_STEPS.SETUP:
        return (
          <ResearchSetup
            formData={formData}
            handleInputChange={handleInputChange}
            fileInputRef={fileInputRef}
            handleFileChange={handleFileChange}
            platform={formData.platform}
            setNetworkData={setNetworkData}
            setOriginalNetworkData={setOriginalNetworkData}
            setFormData={setFormData}
            handleFetchWikipedia={handleFetchWikipedia}
          />
        );
      case ALL_STEPS.WIKIPEDIA:
        return (
          <Discussion
            formData={formData}
            handleInputChange={handleInputChange}
            content={wikiContent}
            selectedSection={selectedSection}
            onSelect={setSelectedSection}
            convertToTxt={async (title) => {
              const result = await fetch(
                `${import.meta.env.VITE_API_URL}/convert-wikipedia-to-txt`,
                {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({
                    filename: "wikipedia_data",
                    section_title: title,
                  }),
                }
              );

              const data = await result.json();
              if (!result.ok)
                throw new Error(data.detail || "Failed to convert to TXT");
              return data;
            }}
          />
        );
      case ALL_STEPS.DATA_CONFIG:
        return (
          <DataConfiguration
            formData={formData}
            handleInputChange={handleInputChange}
          />
        );
      case ALL_STEPS.TIME_FRAME:
        return (
          <TimeFrame
            formData={formData}
            handleInputChange={handleInputChange}
          />
        );
      case ALL_STEPS.MESSAGE_CONTENT:
        return (
          <MessageContent
            formData={formData}
            handleInputChange={handleInputChange}
          />
        );
      case ALL_STEPS.USER_FILTERS:
        return (
          <UserFilters
            formData={formData}
            handleInputChange={handleInputChange}
          />
        );
      case ALL_STEPS.NETWORK_VISUALIZATION:
        return (
          <NetworkVisualization
            networkData={networkData}
            originalNetworkData={originalNetworkData}
            communities={communities}
            communityMap={communityMap}
            setCommunities={setCommunities}
            setCommunityMap={setCommunityMap}
            handleNetworkAnalysis={handleNetworkAnalysis}
            formData={formData}
            setNetworkData={setNetworkData}
            setOriginalNetworkData={setOriginalNetworkData}
            uploadedFileName={formData.uploadedFileName}
            filters={filters}
            setShouldFetchCommunities={setShouldFetchCommunities}
            selectedMetric={selectedMetric}
            setSelectedMetric={setSelectedMetric}
            message={message}
            setMessage={setMessage}
            shouldShowUserFilters={shouldShowUserFilters}
          />
        );
      case ALL_STEPS.COMPARATIVE_ANALYSIS:
        return (
          <ComparativeAnalysis
            originalNetworkData={originalNetworkData}
            comparison={comparison}
            filters={filters}
            uploadedFileName={formData.uploadedFileName}
            platform={formData.platform}
            formData={formData}
            setFormData={setFormData}
            wikiContent={wikiContent}
            setWikiContent={setWikiContent}
            selectedSection={selectedSection}
            setSelectedSection={setSelectedSection}
            handleFetchWikipedia={handleFetchWikipedia}
          />
        );
      case ALL_STEPS.RESEARCH_REPORT:
        return (
          <ResearchReport
            formData={formData}
            networkData={networkData}
            originalNetworkData={originalNetworkData}
            communities={communities}
            selectedMetric={selectedMetric}
            comparison={comparison}
            handleSaveResearch={handleSaveResearch}
          />
        );
      default:
        return <p>Step {currentStep} coming soon...</p>;
    }
  };

  const handleSubmit = () => {
    if (currentUser?.id) {
      setShowSaveModal(true);
    } else {
      setShowLoginInvite(true);
    }
  };

  return (
    <Container fluid className="research-wizard-container">
      <Row className="justify-content-center">
        <Col lg={10} md={12} sm={12}>
          <div className="wizard-header text-center">
            <h2>Network Research</h2>
            <p className="text-muted">
              Configure your network analysis in a few simple steps
            </p>
          </div>

          <div className="wizard-progress-line">
            {getVisibleSteps().map((stepContent, index) => {
              const isCompleted = currentStep > index + 1;
              const isActive = currentStep === index + 1;
              const stepLabel = STEP_LABELS[stepContent];

              return (
                <div
                  key={index}
                  className={`wizard-step ${isCompleted ? "completed" : ""} ${
                    isActive ? "active" : ""
                  }`}
                >
                  <div className="step-circle">{index + 1}</div>
                  <div className="step-line"></div>
                  <div className="step-label">{stepLabel}</div>
                </div>
              );
            })}
          </div>

          {renderCurrentStep()}

          <div className="wizard-navigation mt-4">
            <Button
              variant="outline-secondary"
              className="prev-btn"
              onClick={goToPreviousStep}
              disabled={currentStep === 1}
            >
              <ChevronLeft className="me-1" /> Previous
            </Button>
            {currentStep < getVisibleTotalSteps() ? (
              <Button
                variant="primary"
                className="next-btn"
                onClick={goToNextStep}
              >
                Next <ChevronRight className="ms-1" />
              </Button>
            ) : (
              <Button
                variant="success"
                className="submit-btn"
                onClick={handleSubmit}
              >
                Finish
              </Button>
            )}
          </div>
        </Col>
      </Row>
      <Modal
        show={showSaveModal}
        onHide={() => setShowSaveModal(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Save Research</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Would you like to save your research to your account?
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="secondary"
            onClick={() => {
              setShowSaveModal(false);
              toast.info("Research completed but was not saved.");
            }}
          >
            No, thanks
          </Button>
          <Button
            variant="primary"
            onClick={async () => {
              setShowSaveModal(false);
              try {
                await handleSaveResearch();
                setTimeout(() => navigate("/history"), 800);
              } catch (err) {
                toast.error("Failed to save research.");
              }
            }}
          >
            Yes, save it
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal
        show={showLoginInvite}
        onHide={() => setShowLoginInvite(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Sign Up to Save</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          You completed your research! To save it and manage your research
          history, please sign up or log in.
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowLoginInvite(false)}>
            Not now
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              setShowLoginInvite(false);
              navigate("/signin");
            }}
          >
            Sign Up / Log In
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default ResearchWizard;