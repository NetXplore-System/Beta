import './modal.css'; 

const Modal = ({ children, onClose }) => {
  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog-content" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
};

export default Modal;