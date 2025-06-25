import React from "react";
import { Modal } from "react-bootstrap";

const VideoPopup = ({ show, onHide, videoUrl }) => {
  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton />
      <Modal.Body style={{ padding: 0 }}>
        <div
          className="video-container"
          style={{ position: "relative", paddingBottom: "56.25%", height: 0 }}
        >
          <iframe
            src={videoUrl}
            title="Demo Video"
            frameBorder="0"
            allowFullScreen
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
            }}
          ></iframe>
        </div>
      </Modal.Body>
    </Modal>
  );
};

export default VideoPopup;
