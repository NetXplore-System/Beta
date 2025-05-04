import React from "react";
import { Row, Col, Card, Button, Form } from "react-bootstrap";
import {  Save, Trash } from "react-bootstrap-icons";
import AnonymizationToggle from "../AnonymizationToggle.jsx";
import FileUploader from "../common/FileUploader.jsx";
import MyResearchReport from "../utils/MadeReport.jsx";

const ResearchCard = ({
  name,
  setName,
  description,
  setDescription,
  isAnonymized,
  setIsAnonymized,
  handleUploadClick,
  handleDelete,
  fileInputRef,
  handleFileChange,
  inputKey,
  message,
  handleSave,
  showDownload,
  setShowDownload,
  networkData,
  filters,
  selectedMetric,
  hasComparison,
}) => {

  return (
    <Card className="research-card">
      <Form>
        <Row className="align-items-center justify-content-between">
          <Col>
            <h4 className="fw-bold">New Research</h4>
          </Col>
          <Col className="text-end">

            {showDownload && (
              <MyResearchReport
                hasComparison={hasComparison}
                selectedMetric={selectedMetric}
                name={name}
                description={description}
                params={filters.buildNetworkFilterParams()}
                setShowDownload={setShowDownload} 
                />
            )}
            {networkData && (
              <Button
                className="action-btn me-2"
                onClick={() => setShowDownload(!showDownload)}
              >
                {/* <Download size={16} /> */}
                Export Report
              </Button>
            )}
            <Button onClick={handleSave} className="action-btn me-2">
              <Save size={16} /> Save
            </Button>
            <Button onClick={handleDelete} className="action-btn delete-btn">
              <Trash size={16} /> Delete File
            </Button>
          </Col>
        </Row>
        <Row className="mt-3 align-items-center">
          <Col lg={8} md={12}>
            <Form.Group className="mb-3">
              <Form.Label className="research-label">Research Name:</Form.Label>
              <Form.Control
                type="text"
                value={name}
                placeholder="Enter the name of your research"
                onChange={(e) => setName(e.target.value)}
                className="research-input"
              />
            </Form.Group>
            <Form.Group>
              <Form.Label className="research-label">Description:</Form.Label>
              <Form.Control
                type="text"
                value={description}
                placeholder="Enter a short description"
                onChange={(e) => setDescription(e.target.value)}
                className="research-input"
              />
            </Form.Group>
            <div className="mx-2 my-4 align-items-center">
              <AnonymizationToggle
                isAnonymized={isAnonymized}
                setIsAnonymized={setIsAnonymized}
              />
            </div>
          </Col>
          <Col
            lg={4}
            md={12}
            className="d-flex flex-column align-items-center mt-3 mt-lg-0"
          >
            <FileUploader
              inputKey={inputKey}
              fileInputRef={fileInputRef}
              handleUploadClick={handleUploadClick}
              handleFileChange={handleFileChange}
              message={message}
            />
          </Col>
        </Row>
      </Form>
    </Card>
  );
};

export default ResearchCard;
