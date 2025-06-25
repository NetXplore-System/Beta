import { useState, useEffect, useMemo } from "react";
import { useSelector } from "react-redux";
import { useLocation } from "react-router-dom";
import { toast } from "sonner";
import {
  FaEye,
  FaEdit,
  FaExchangeAlt,
  FaTrash,
  FaDownload,
  FaSort,
  FaSortUp,
  FaSortDown,
  FaSearch,
} from "react-icons/fa";
import {
  Button,
  ButtonGroup,
  Card,
  Table,
  Form,
  InputGroup,
  Row,
  Col,
  Badge,
} from "react-bootstrap";

import Loader from "../components/utils/Loader";
import Modal from "../components/utils/Modal";
import ResearchHistory from "../components/utils/ResearcHistoryComp";
import UpdateResearch from "../components/utils/UpdateResearch";
import ComparisonHistory from "../components/utils/HistoryComparison";

import "../components/utils/history.css";
import { deleteResearch } from "../components/utils/ApiService";

const History = () => {
  const user = useSelector((state) => state.user);
  const [userHistory, setUserHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const viewResearchId = queryParams.get("view");
  const [research, setResearch] = useState(null);
  const [action, setAction] = useState({
    inAction: false,
    ids: [],
  });
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState({
    key: null,
    direction: "asc",
  });
  const [filterPlatform, setFilterPlatform] = useState("");

  const resetAction = (id) => {
    setAction((prev) => ({
      inAction: !prev.inAction,
      ids: prev.ids.filter((val) => val != id),
    }));
  };

  const handleDelete = async (researchId) => {
    setAction((prev) => ({
      inAction: true,
      ids: [...prev.ids, researchId],
    }));

    toast.promise(
      deleteResearch(researchId, user?.token),
      {
        loading: "Deleting research...",
        success: (data) => {
          setUserHistory((prev) =>
            prev.filter((research) => research.id !== researchId)
          );
          resetAction(researchId);
          return data;
        },
        error: (data) => {
          resetAction(researchId);
          return data;
        },
      },
      {
        closeButton: true,
      }
    );
  };

  const updateResearchs = (researchData) => {
    setUserHistory(
      userHistory.map((research) =>
        research.id === researchData.id
          ? { ...research, ...researchData }
          : research
      )
    );
  };

  const closeModal = () => {
    setResearch(null);
  };

  useEffect(() => {
    async function getUserHistory() {
      if (!user?.currentUser?.id) return;
      try {
        setLoading(true);
        const history = await fetch(
          `${import.meta.env.VITE_API_URL}/history/${user.currentUser.id}`,
          {
            headers: {
              Authorization: `Bearer ${user?.token}`,
            },
          }
        );
        if (!history.ok) {
          const { detail } = await history.json();
          toast.error("Error fetching user history");
          return;
        }

        const data = await history.json();
        if (!data.history.length) {
          toast.error("Don't find history. Please create research");
          return;
        }
        setUserHistory(data.history);
        if (viewResearchId) {
          const target = data.history.find(
            (item) => item.id === viewResearchId
          );
          if (target) {
            setResearch({ ...target, button: "view" });
          }
        }
      } catch (error) {
        toast.error("Error fetching user history");
      } finally {
        setLoading(false);
      }
    }

    getUserHistory();
  }, [user?.currentUser?.id]);

  const handleSort = (key) => {
    let direction = "asc";
    if (sortConfig.key === key && sortConfig.direction === "asc") {
      direction = "desc";
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) {
      return (
        <FaSort className="history-sort-icon history-sort-icon-inactive" />
      );
    }
    return sortConfig.direction === "asc" ? (
      <FaSortUp className="history-sort-icon" />
    ) : (
      <FaSortDown className="history-sort-icon" />
    );
  };

  const uniquePlatforms = useMemo(() => {
    const platforms = [...new Set(userHistory.map((item) => item.platform))];
    return platforms.filter(Boolean);
  }, [userHistory]);

  const filteredAndSortedData = useMemo(() => {
    let filtered = userHistory;

    if (searchTerm) {
      filtered = filtered.filter(
        (item) =>
          item.research_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          item.platform.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterPlatform) {
      filtered = filtered.filter((item) => item.platform === filterPlatform);
    }

    if (sortConfig.key) {
      filtered = [...filtered].sort((a, b) => {
        let aValue = a[sortConfig.key];
        let bValue = b[sortConfig.key];

        if (sortConfig.key === "created_at") {
          aValue = new Date(aValue);
          bValue = new Date(bValue);
        }

        if (typeof aValue === "string" && typeof bValue === "string") {
          aValue = aValue.toLowerCase();
          bValue = bValue.toLowerCase();
        }

        if (aValue < bValue) {
          return sortConfig.direction === "asc" ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === "asc" ? 1 : -1;
        }
        return 0;
      });
    }

    return filtered;
  }, [userHistory, searchTerm, filterPlatform, sortConfig]);

  const clearFilters = () => {
    setSearchTerm("");
    setFilterPlatform("");
    setSortConfig({ key: null, direction: "asc" });
  };

  const getHistoryStats = () => {
    const totalResearch = userHistory.length;
    const platformsCount = uniquePlatforms.length;
    const recentCount = userHistory.filter((item) => {
      const created = new Date(item.created_at);
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      return created > oneWeekAgo;
    }).length;

    return { totalResearch, platformsCount, recentCount };
  };

  const stats = getHistoryStats();

  const handleDownloadCSV = async (researchId) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/export/excel/${researchId}`,
        {
          headers: {
            Authorization: `Bearer ${user?.token}`,
          },
        }
      );

      if (!response.ok) {
        const data = await response.json();
        toast.error(data.detail || "Failed to export CSV");
        return;
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `research_${researchId}.xlsx`);

      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      toast.error("An error occurred while downloading CSV");
      console.error(error);
    }
  };

  return (
    <div className="history-container">
      {research && (
        <Modal onClose={closeModal} showCloseButton={true}>
          {research.button === "view" && (
            <ResearchHistory research={research} />
          )}
          {research.button === "edit" && (
            <UpdateResearch
              research={research}
              setResearch={setResearch}
              updateResearchs={updateResearchs}
            />
          )}
          {research.button === "compare" && (
            <ComparisonHistory research={research} />
          )}
        </Modal>
      )}
      <Card className={`history-table mt-4 ${loading ? "h-75" : ""}`}>
        <Card.Header>
          <div className="d-flex justify-content-between align-items-center">
            <h4 className="m-0 fw-bold">Research History</h4>
            {userHistory.length > 0 && (
              <div className="text-end">
                <small className="history-date-range">
                  {`${new Date(
                    userHistory?.at(-1)?.created_at
                  ).toLocaleDateString()} - ${new Date(
                    userHistory?.at(0)?.created_at
                  ).toLocaleDateString()}`}
                </small>
              </div>
            )}
          </div>
        </Card.Header>

        {!loading && userHistory.length > 0 && (
          <Card.Body className="border-bottom history-stats-summary">
            <Row>
              <Col md={3}>
                <div className="history-stat-item">
                  <h5>Total Research</h5>
                  <h2>{stats.totalResearch}</h2>
                </div>
              </Col>
              <Col md={3}>
                <div className="history-stat-item">
                  <h5>Platforms</h5>
                  <h2>{stats.platformsCount}</h2>
                </div>
              </Col>
              <Col md={3}>
                <div className="history-stat-item">
                  <h5>This Week</h5>
                  <h2>{stats.recentCount}</h2>
                </div>
              </Col>
              <Col md={3}>
                <div className="history-stat-item">
                  <h5>Showing</h5>
                  <h2>{filteredAndSortedData.length}</h2>
                </div>
              </Col>
            </Row>
          </Card.Body>
        )}

        {!loading && userHistory.length > 0 && (
          <Card.Body className="border-bottom">
            <Row className="g-3 align-items-end">
              <Col md={5}>
                <Form.Label className="history-filter-label">Search</Form.Label>
                <InputGroup>
                  <InputGroup.Text className="history-search-icon">
                    <FaSearch />
                  </InputGroup.Text>
                  <Form.Control
                    type="text"
                    placeholder="Search by name or platform..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="history-search-input"
                  />
                </InputGroup>
              </Col>
              <Col md={3}>
                <Form.Label className="history-filter-label">
                  Platform Filter
                </Form.Label>
                <Form.Select
                  value={filterPlatform}
                  onChange={(e) => setFilterPlatform(e.target.value)}
                  className="history-filter-select"
                >
                  <option value="">All Platforms</option>
                  {uniquePlatforms.map((platform) => (
                    <option key={platform} value={platform}>
                      {platform}
                    </option>
                  ))}
                </Form.Select>
              </Col>
              <Col md={4}>
                <div className="d-flex justify-content-end gap-2">
                  <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={clearFilters}
                    disabled={!searchTerm && !filterPlatform && !sortConfig.key}
                    className="history-clear-filters-btn"
                  >
                    Clear Filters
                  </Button>
                </div>
              </Col>
            </Row>
          </Card.Body>
        )}

        <Card.Body>
          {loading ? (
            <div className="history-loader-container">
              <Loader />
            </div>
          ) : filteredAndSortedData.length === 0 ? (
            <div className="history-empty-state">
              {userHistory.length === 0 ? (
                <>
                  <h5>No Research History</h5>
                  <p>Create your first research to see it here.</p>
                </>
              ) : (
                <>
                  <h5>No Results Found</h5>
                  <p>No research matches your search criteria.</p>
                  <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={clearFilters}
                    className="history-clear-filters-btn"
                  >
                    Clear Filters
                  </Button>
                </>
              )}
            </div>
          ) : (
            <div className="table-responsive">
              <Table className="table">
                <thead>
                  <tr>
                    <th onClick={() => handleSort("research_name")}>
                      Research Name {getSortIcon("research_name")}
                    </th>
                    <th onClick={() => handleSort("created_at")}>
                      Date Created {getSortIcon("created_at")}
                    </th>
                    <th onClick={() => handleSort("platform")}>
                      Platform {getSortIcon("platform")}
                    </th>
                    <th>Actions</th>
                    <th>Export</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSortedData.map((research) => (
                    <tr key={research.id}>
                      <td>
                        <div className="history-research-name">
                          {research.research_name}
                        </div>
                      </td>
                      <td>
                        <span className="history-date-text">
                          {new Date(research.created_at).toLocaleDateString()}
                        </span>
                      </td>
                      <td>
                        <Badge className="history-platform-badge">
                          {research.platform}
                        </Badge>
                      </td>
                      <td>
                        <div className="history-action-buttons">
                          <ButtonGroup size="sm">
                            <Button
                              variant="primary"
                              data-tooltip-id="my-tooltip"
                              data-tooltip-content="View research details"
                              data-tooltip-place="top"
                              aria-label="View details"
                              onClick={() =>
                                setResearch({ ...research, button: "view" })
                              }
                            >
                              <FaEye />
                            </Button>
                            <Button
                              variant="primary"
                              data-tooltip-id="my-tooltip"
                              data-tooltip-content="Edit research"
                              data-tooltip-place="top"
                              aria-label="Edit"
                              onClick={() =>
                                setResearch({ ...research, button: "edit" })
                              }
                            >
                              <FaEdit />
                            </Button>
                            <Button
                              variant="primary"
                              data-tooltip-id="my-tooltip"
                              data-tooltip-content="Compare research"
                              data-tooltip-place="top"
                              aria-label="Compare"
                              onClick={() =>
                                setResearch({ ...research, button: "compare" })
                              }
                            >
                              <FaExchangeAlt />
                            </Button>
                            <Button
                              variant="primary"
                              data-tooltip-id="my-tooltip"
                              data-tooltip-content="Delete research"
                              data-tooltip-place="top"
                              aria-label="Delete"
                              onClick={() => handleDelete(research.id)}
                              disabled={
                                action.inAction &&
                                action.ids.some((id) => id === research.id)
                              }
                            >
                              <FaTrash />
                            </Button>
                          </ButtonGroup>
                        </div>
                      </td>
                      <td>
                        <Button
                          variant="primary"
                          data-tooltip-id="my-tooltip"
                          data-tooltip-content="Download CSV"
                          data-tooltip-place="top"
                          size="sm"
                          onClick={() => handleDownloadCSV(research.id)}
                          aria-label="Download CSV"
                        >
                          <FaDownload />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default History;
