import { useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useSelector } from 'react-redux';
import { Button, Card, Form } from 'react-bootstrap';
import { AiOutlineLoading } from 'react-icons/ai';



export default function UpdateResearch({ research, setResearch, updateResearchs }) {
  const user = useSelector((state) => state.user);
  const [researchData, setResearchData] = useState({
    research_name: research.research_name || '',
    description: research.description || '',
    filters: {
      start_date: research.filters?.start_date || '',
      end_date: research.filters?.end_date || '',
      start_time: research.filters?.start_time || '',
      end_time: research.filters?.end_time || '',
      message_limit: research.filters?.message_limit || '',
      limit_type: research.filters?.limit_type || 'first',
      min_message_length: research.filters?.min_message_length || '',
      max_message_length: research.filters?.max_message_length || '',
      keywords: research.filters?.keywords || '',
      min_messages: research.filters?.min_messages || '',
      max_messages: research.filters?.max_messages || '',
      top_active_users: research.filters?.top_active_users || '',
      selected_users: research.filters?.selected_users || '',
      filter_by_username: research.filters?.filter_by_username || '',
      anonymize: research.filters?.anonymize || false,
      algorithm: research.filters?.algorithm || 'louvain',
    },
  });
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    setResearchData((prevData) => {
      if (name in prevData.filters) {
        return {
          ...prevData,
          filters: {
            ...prevData.filters,
            [name]: newValue,
          },
        };
      } else {
        return {
          ...prevData,
          [name]: newValue,
        };
      }
    });
  };

  const handleCancel = () => {
    setResearch(null);
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      if (!file) {
        toast.error('Please upload the file the you used for the research');
        return;
      }
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${import.meta.env.VITE_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        toast.error('Error uploading file');
        console.error('Error uploading file:', res);
        return;
      }
      const response = await fetch(`${import.meta.env.VITE_API_URL}/research/${research.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`,
        },
        body: JSON.stringify({ ...researchData, file_name: file.name }),
      });
      if (response.ok) {
        setResearch(researchData);
        setResearch(null);
        const data = await response.json();
        updateResearchs(data.data);
        toast.success('Research data saved successfully');
      } else {
        toast.error('Error saving research data');
        console.error('Error saving research data:', response);
      }
    } catch (error) {
      toast.error('Error saving research data');
      console.error('Error saving research data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleFileClick = (e) => {
    e.stopPropagation();
    e.preventDefault();
    fileRef.current.click();
  }

  return (
    <Card className='update-history'>
      <Card.Header>
        <h5 className='fw-bold'>Update Research</h5>
      </Card.Header>
      <Card.Body>
          <div className='row'>
            <Form.Group className='column full-width'>
              <Form.Label className="research-label">Research Name:</Form.Label>
              <Form.Control
                type="text"
                name="research_name"
                value={researchData.research_name}
                onChange={handleChange}
              />
            </Form.Group>
            <Form.Group className='column full-width'>
              <Form.Label className="research-label">Description:</Form.Label>
              <textarea
                className='form-control'
                name="description"
                style={{ width: '100%' }}
                rows={3}
                value={researchData.description}
                onChange={handleChange}
              />
            </Form.Group>
          </div>
          <div className='row'>
            {Object.keys(researchData.filters).map((key) => (
              key === 'anonymize' ?
                null
                :
                <Form.Group className='column' key={key}>
                  <Form.Label className="research-label">{key.replace(/_/g, ' ')}:</Form.Label>
                  <Form.Control
                    type={(key.split("_")?.[1] === 'date' || key.split("_")?.[1] === 'time') ?
                      key.split("_")[1] === 'date' ? 'date' : 'time'
                      :
                      'text'}
                    name={key}
                    value={researchData.filters[key]}
                    onChange={handleChange}
                  />
                </Form.Group>
            ))}
            <Form.Check
              type={'checkbox'}
              label={'anonymize'}
              checked={researchData.filters['anonymize']}
              name={'anonymize'}
              onChange={handleChange}
              className='column input-checkbox'
            />
            <div className='column full-width' >
              <button className='generic-button' onClick={handleFileClick}>Upload File</button>
              <span className='file-name'>{file ? file.name : 'No file selected'}</span>
              <p className='file-description'>Please upload the file you used for the research</p>
              <input className='d-none' ref={fileRef} type="file" onChange={handleFileChange} />
            </div>
          </div>
          <div className="d-flex justify-content-end mt-3 gap-2">
            <Button variant='outline-danger' onClick={handleCancel} >
              Cancel
            </Button>
            <Button className='generic-button' onClick={handleSave} disabled={loading} >
              {loading ? <><AiOutlineLoading className="spinner-icon" /> Saving...</> : 'Save'}
            </Button>
          </div>
      </Card.Body>
    </Card>
  );
}