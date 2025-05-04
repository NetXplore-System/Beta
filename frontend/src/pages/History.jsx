import { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { toast } from 'react-hot-toast';
import Loader from '../components/utils/Loader';
import { FaEye, FaEdit, FaCopy, FaTrash } from 'react-icons/fa';
import Modal from '../components/utils/Modal';
import ResearchHistory from '../components/utils/ResearcHistory';
import UpdateResearch from '../components/utils/UpdateResearch';
import ComparisonHistory from '../components/utils/HistoryComparison';
import { Badge, Button, ButtonGroup, Card, Table } from 'react-bootstrap';
import { Tooltip } from 'react-tooltip'


import "../components/utils/history.css";

const History = () => {
    const user = useSelector(state => state.user);
    const [userHistory, setUserHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [research, setResearch] = useState(null);
    const [inAction, setInAction] = useState(false);

    const handleDelete = async (researchId) => {
        try {
            setInAction(true);
            const response = await fetch(`${import.meta.env.VITE_API_URL}/research/${researchId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${user?.token}`
                }
            });
            if (response.ok) {
                toast.success('Research deleted successfully');
                setUserHistory(userHistory.filter(research => research.id !== researchId));
            } else {
                toast.error('Failed to delete research');
            }
        } catch (error) {
            console.error('Error deleting research:', error);
            toast.error('Error deleting research');
        } finally {
            setInAction(false);
        }
    };

    const updateResearchs = (researchData) => {
        setUserHistory(userHistory.map(research => research.id === researchData.id ? { ...research, ...researchData } : research));
    };

    useEffect(() => {
        async function getUserHistory() {
            try {
                setLoading(true);
                const history = await fetch(`${import.meta.env.VITE_API_URL}/history/${user?.currentUser?.id}`, {
                    headers: {
                        'Authorization': `Bearer ${user?.token}`
                    }
                });
                const data = await history.json();
                if (!data.history.length) {
                    toast.error('don\'t find history. please create research');
                    return;
                }
                setUserHistory(data.history);
                toast.success('User history fetched successfully');
            } catch (error) {
                console.error('Error fetching user history:', error);
                toast.error('Error fetching user history');
            } finally {
                setLoading(false);
            }
        }
        getUserHistory();
    }, []);

    return (
        <div className='history-container'>
            <Tooltip id="my-tooltip" />
                    {research && (
                        <Modal onClose={() => setResearch(null)}>
                            {research.button === 'view' && <ResearchHistory research={research} />}
                            {research.button === 'edit' && <UpdateResearch research={research} setResearch={setResearch} updateResearchs={updateResearchs} />}
                            {research.button === 'compare' && <ComparisonHistory research={research} />}
                        </Modal>
                    )}
            <Card className="history-table mt-4">
                <Card.Header className="d-flex justify-content-between align-items-center">
                    <h4 className="m-0 fw-bold">History Area</h4>
                    {userHistory.length ? <p className="m-0 fw-bold">Date Rang: {`${new Date(userHistory?.at(0)?.created_at).toLocaleDateString()} - ${new Date(userHistory?.at(-1)?.created_at).toLocaleDateString()}`}</p> : ""}
                </Card.Header>
                <Card.Body>
                    {loading ? (
                        <div className="loader-container">
                            <Loader />
                        </div>
                    ) : (
                        <Table striped bordered hover>
                            <thead>
                                <tr>
                                    <th>Research Name</th>
                                    <th>Date Created</th>
                                    <th>Platform</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {userHistory.map((research) => (
                                    <tr key={research.id}>
                                        <td>{research.research_name}</td>
                                        <td>{new Date(research.created_at).toLocaleDateString()}</td>
                                        <td>{research.platform}</td>
                                        <td>
                                            <Badge bg="info" className="community-badge">
                                                {research.status || 'N/A'}
                                            </Badge>
                                           
                                        </td>
                                        <td>
                                            <ButtonGroup size='sm'>
                                                <Button data-tooltip-id="my-tooltip"  data-tooltip-content="<your placeholder>" aria-label="View details" onClick={() => setResearch({ ...research, button: 'view' })}>
                                                    <FaEye />
                                                </Button>
                                                <Button  onClick={() => setResearch({ ...research, button: 'edit' })} aria-label="Edit">
                                                    <FaEdit />
                                                </Button>
                                                <Button  onClick={() => setResearch({ ...research, button: 'compare' })} aria-label="Compare">
                                                    <FaCopy />
                                                </Button>
                                                <Button  onClick={() => handleDelete(research.id)} aria-label="Delete" disabled={inAction}>
                                                    <FaTrash />
                                                </Button>
                                            </ButtonGroup>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </Table>
                        /* </table> */
                    )}
                </Card.Body>
            </Card>
        </div>
    );
};

export default History;