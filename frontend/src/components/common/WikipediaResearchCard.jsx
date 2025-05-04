import React from "react";
import { Row, Col, Card, Button, Form } from "react-bootstrap";
import { Save, Trash, Link45deg } from "react-bootstrap-icons";
import { AlertBox } from "../../pages/Form.style";
import MyResearchReport from "../utils/ResearchReport.jsx";

const WikipediaResearchCard = ({
  name,
  setName,
  description,
  setDescription,
  wikiUrl,
  setWikiUrl,
  handleWikipediaSubmit,
  handleDelete,
  showDownload,
  setShowDownload,
  networkData,
  filters,
  selectedMetric,
  hasComparison,
  handleSave,
  message,
}) => {
  return (
    <Card className="research-card">
      <Form>
        <Row className="align-items-center justify-content-between">
          <Col>
            <h4 className="fw-bold">New Wikipedia Research</h4>
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
                Export Report
              </Button>
            )}
            <Button onClick={handleSave} className="action-btn me-2">
              <Save size={16} /> Save
            </Button>
            <Button onClick={handleDelete} className="action-btn delete-btn">
              <Trash size={16} /> Delete
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
            <Form.Group className="mt-4">
              <Form.Label className="research-label">Wikipedia Discussion URL:</Form.Label>
              <div className="d-flex">
                <Form.Control
                  type="text"
                  value={wikiUrl}
                  placeholder="Paste Wikipedia discussion link here"
                  onChange={(e) => setWikiUrl(e.target.value)}
                  className="research-input me-2"
                />
                <Button onClick={handleWikipediaSubmit} className="action-btn">
                  <Link45deg size={18} /> Upload
                </Button>
              </div>
            </Form.Group>
          </Col>
        </Row>
        {message && (
          <Row className="mt-3">
            <Col>
              <AlertBox>{message}</AlertBox>
            </Col>
          </Row>
        )}
      </Form>
    </Card>
  );
};

export default WikipediaResearchCard;
