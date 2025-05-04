import React, { useState } from "react";
import { Card, Row, Col, Form, Button } from "react-bootstrap";
import DateRangeFilter from "./DateRangeFilter";
import UserFilter from "./UserFilter";
import MessageFilter from "./MessageFilter";
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight } from "react-bootstrap-icons";
import './FilterForm.css';



const FilterForm = ({
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
  isAnonymized,
  setIsAnonymized,
  handleNetworkAnalysis,
}) => {
  const [showFilters, setShowFilters] = useState(true);
  const [currentStep, setCurrentStep] = useState(1);

  const totalSteps = 4;

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
  };

  const nextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  
  const resetFilters = () => {
    setStartDate("");
    setEndDate("");
    setStartTime("");
    setEndTime("");
    setMessageLimit(50);
    setLimitType("first");
    setMinMessageLength(10);
    setMaxMessageLength(100);
    setKeywords("");
    setUsernameFilter("");
    setMinMessages("");
    setMaxMessages("");
    setActiveUsers("");
    setSelectedUsers([]);
    setCurrentStep(1);
  };

  const renderStepIndicator = () => {
    const steps = [
      { number: 1, label: "Time Frame" },
      { number: 2, label: "Message Criteria" },
      { number: 3, label: "User Filters" },
      { number: 4, label: "Review" }
    ];
    
    const progressWidth = `${((currentStep - 1) / (totalSteps - 1)) * 100}%`;

    return (
      <div className="step-indicator">
        <div className="progress-line" style={{ width: progressWidth }}></div>
        
        {steps.map((step) => (
          <div className="step" key={step.number}>
            <div 
              className={`step-bubble ${
                currentStep > step.number 
                  ? 'completed' 
                  : currentStep === step.number 
                  ? 'active' 
                  : ''
              }`}
            >
              {step.number}
            </div>
            <div 
              className={`step-label ${currentStep === step.number ? 'active' : ''}`}
            >
              {step.label}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <DateRangeFilter
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
            handleInputChange={handleInputChange}
          />
        );
      case 2:
        return (
          <MessageFilter
            minMessageLength={minMessageLength}
            setMinMessageLength={setMinMessageLength}
            maxMessageLength={maxMessageLength}
            setMaxMessageLength={setMaxMessageLength}
            keywords={keywords}
            setKeywords={setKeywords}
            handleInputChange={handleInputChange}
          />
        );
      case 3:
        return (
          <UserFilter
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
          />
        );
      case 4:
        return (
          <div className="summary-section">
            <div className="summary-group">
              <h6>Time Frame</h6>
              <div className="summary-item">
                <div className="summary-label">Date Range:</div>
                <div className="summary-value">{startDate || 'Any'} to {endDate || 'Any'}</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Time Range:</div>
                <div className="summary-value">{startTime || 'Any'} to {endTime || 'Any'}</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Message Count:</div>
                <div className="summary-value">{messageLimit} {limitType === 'first' ? 'first' : 'last'} messages</div>
              </div>
            </div>
            
            <div className="summary-group">
              <h6>Message Criteria</h6>
              <div className="summary-item">
                <div className="summary-label">Length Range:</div>
                <div className="summary-value">{minMessageLength} to {maxMessageLength} characters</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Keywords:</div>
                <div className="summary-value">{keywords || 'None'}</div>
              </div>
            </div>
            
            <div className="summary-group">
              <h6>User Filters</h6>
              <div className="summary-item">
                <div className="summary-label">Username Filter:</div>
                <div className="summary-value">{usernameFilter || 'None'}</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Message Count:</div>
                <div className="summary-value">Min: {minMessages || 'Any'}, Max: {maxMessages || 'Any'}</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Top Active Users:</div>
                <div className="summary-value">{activeUsers || 'None'}</div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Selected Users:</div>
                <div className="summary-value">{selectedUsers ? selectedUsers : 'None'}</div>
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <Card className="research-card">
      <h4 className="fw-bold d-flex justify-content-between align-items-center mb-4">
        Research Filters
        <Button
          variant="link"
          className="toggle-btn p-0"
          onClick={() => setShowFilters(!showFilters)}
        >
          {showFilters ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </Button>
      </h4>
      
      {showFilters && (
        <div>
          {renderStepIndicator()}
          
          <div className="filter-content">
            {renderCurrentStep()}
          </div>
          
          <div className="navigation-buttons">
            <div>
              {currentStep > 1 && (
                <Button 
                  variant="outline-primary" 
                  onClick={prevStep}
                  className="btn-prev"
                >
                  <ChevronLeft size={16} className="me-1" /> Back
                </Button>
              )}
            </div>
            
            <div className="d-flex">
              <Button 
                variant="outline-danger" 
                onClick={resetFilters}
                className="btn-reset me-2"
              >
                Reset All
              </Button>
              
              {currentStep < totalSteps ? (
                <Button 
                  variant="primary" 
                  onClick={nextStep}
                  className="btn-next"
                >
                  Next <ChevronRight size={16} className="ms-1" />
                </Button>
              ) : (
                <Button 
                  variant="success" 
                  onClick={handleNetworkAnalysis}
                  className="btn-apply"
                >
                  Apply Filters
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};
export default FilterForm;
