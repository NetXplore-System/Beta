import React from "react";
import { Card } from "react-bootstrap";
import DiscussionSectionPicker from "../../components/filters/DiscussionSectionPicker";

const Discussion = ({
  content,
  onSelect,
}) => {
  return (
    <Card className="research-card">
      <Card.Body>
        <h3 className="step-title">Discussion Sections</h3>
        <DiscussionSectionPicker
          content={content}
          onSelect={onSelect}
        />
      </Card.Body>
    </Card>
  );
};

export default Discussion;
