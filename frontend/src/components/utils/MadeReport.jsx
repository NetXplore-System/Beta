import { useState } from 'react';
import { useSelector } from 'react-redux';
import Modal from './Modal';
import { PDFDownloadLink } from '@react-pdf/renderer';
import Report, { formatFilterLabel } from './Report';
import './made-report.css';

const MadeReport = ({ selectedMetric, name, params, setShowDownload, hasComparison }) => {
    const { tableData } = useSelector((state) => state.table);
    const { main, comparison: { source, comparison } } = useSelector((state) => state.images);
    const { currentUser } = useSelector((state) => state.user);

    const [research, setResearch] = useState({
        name: name || 'unknown',
        researcherName: currentUser?.name || 'unknown',
        date: new Date().toISOString().split('T')[0],
        filters: params ? Array.from(params).map(([key, value]) => `${key}: ${value}`) : [],
        conclusion: '',
        hasComparison,
        metric: selectedMetric,
        stats: tableData || [],
    });
    const [show, setShow] = useState({
        images: [
            ...main.map((data) => ({ ...data, selected: true, description: '' })),
            ...source.map((data) => ({ ...data, selected: true, description: '' })),
            ...comparison.map((data) => ({ ...data, selected: true, description: '' })),
        ],
        filters: research.filters.map((_, index) => ({ index, selected: true })),
    });

    const handleCange = (e, index) => {
        e.stopPropagation();
        const updatedImages = [...show.images];
        updatedImages[index].description = e.target.value;
        setShow((prev) => ({ ...prev, images: updatedImages }));
    }

    const handleToggle = (index, key) => {
        setShow((prev) => {
            const updated = [...prev[key]];
            updated[index].selected = !updated[index].selected;
            return { ...prev, [key]: updated };
        });
    }
    const closeModal = (e) => { e.stopPropagation(); setShowDownload(false) };

    return (
        <Modal onClose={closeModal}>
            <div className="research-report-header" >
                <h2>Research Report</h2>
                <button className="close-btn" onClick={closeModal}>×</button>
            </div>

            <div className="research-report-info" >
                <h3>Research Information</h3>
                <p>Research Name: {research.name}</p>
                <p>Researcher Name: {research.researcherName}</p>
            </div>
            <div className="research-report-images">
                <h3>Include Images</h3>
                {show.images.map((image, index) => (
                    <div key={index} className="image-selection">
                        <label>
                            <img src={image.data} alt={`Graph ${index + 1}`} style={{ height: '100px', width: '100px' }} />
                            <input
                                type="checkbox"
                                checked={show.images[index]?.selected}
                                onChange={() => handleToggle(index, 'images')}
                            />
                            {show.images[index]?.selected ? 'Include this image' : 'Exclude this image'}
                        </label>
                        <input
                            type="text"
                            placeholder="Image description"
                            maxLength={60}
                            value={show.images[index]?.description}
                            onChange={e => handleCange(e, index)}
                        />
                    </div>
                ))}
                {show.images.length === 0 && <p>No images available to include.</p>}
            </div>

            <div className="research-report-filters">
                <h3>Include Filters</h3>
                {research.filters.map((filter, index) => (
                    <div key={index} className="filter-selection">
                        <label>
                            <input
                                type="checkbox"
                                checked={show.filters[index]?.selected}
                                onChange={() => handleToggle(index, 'filters')}
                            />
                            {formatFilterLabel(filter)}
                        </label>
                    </div>
                ))}
                {research.filters.length === 0 && <p>No filters available to include.</p>}
            </div>
            <form className="research-report-form" >

                <div>
                    <label htmlFor="conclusion">Research Conclusion</label>
                    <textarea
                        id="conclusion"
                        value={research.conclusion}
                        onChange={(e) => setResearch({ ...research, conclusion: e.target.value })}
                        placeholder="Enter your research conclusion or findings"
                    />
                </div>

                <div className="checkbox-group">
                    <input
                        type="checkbox"
                        id="hasComparison"
                        checked={research.hasComparison}
                        onChange={(e) => setResearch({ ...research, hasComparison: e.target.checked })}
                    />
                    <label htmlFor="hasComparison">This research includes comparison with other studies</label>
                </div>

                <div className="research-report-actions">
                    <button type="button" className="close-btn" onClick={closeModal}>
                        Cancel
                    </button>

                    {research.name && research.researcherName && research.conclusion && (
                        <PDFDownloadLink
                            document={<Report research={research} show={show} />}
                            fileName="research_report.pdf"
                            className="download-link"
                            onClick={(e) => {
                                setTimeout(() => {
                                    e.stopPropagation();
                                    setShowDownload(false);
                                }, 1000);
                            }}
                        >
                            {({ loading }) => (loading ? 'Preparing...' : 'Download Report')}
                        </PDFDownloadLink>
                    )}
                </div>
            </form>
        </Modal>
    );
};
export default MadeReport;