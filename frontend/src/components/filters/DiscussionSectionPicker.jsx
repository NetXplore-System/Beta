import React, { useState, useEffect } from "react";
import "../../styles/DiscussionSectionPicker.css";

const DiscussionSectionPicker = ({
  content,
  onSelect,
}) => {
  const [selectedTitle, setSelectedTitle] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [minComments, setMinComments] = useState(0);
  const [sortOrder, setSortOrder] = useState("desc");


  const handleSelect = async (section) => {
    setSelectedTitle(section.title);
    onSelect(section);
  };

  const filteredOptions = content?.filter((section) => {
    const titleMatch = section.title
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const commentMatch = section.len_comments >= minComments;
    return titleMatch && commentMatch;
  });

  const sortedOptions = [...filteredOptions].sort((a, b) => {
    return sortOrder === "asc"
      ? a.len_comments - b.len_comments
      : b.len_comments - a.len_comments;
  });

  const totalComments = sortedOptions.reduce(
    (acc, section) => acc + section.len_comments,
    0
  );

  return (
    <div className="discussion-section-picker">
      {sortedOptions.length > 0 && (
        <div className="section-stats-row">
          <div className="stats-label">
            Found: <span className="stats-number">{sortedOptions.length}</span>{" "}
            sections with <span className="stats-number">{totalComments}</span>{" "}
            comments
          </div>
          <div className="filters-inline">
            <input
              type="text"
              placeholder="Search by title..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="filter-input"
            />
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
              className="filter-select"
            >
              <option value="desc">Most comments</option>
              <option value="asc">Least comments</option>
            </select>
          </div>
        </div>
      )}

      {sortedOptions.length > 0 ? (
        <div className="sections-grid">
          {sortedOptions.map((section, idx) => (
            <div
              key={idx}
              className={`section-card ${
                selectedTitle === section.title ? "section-card-selected" : ""
              }`}
              onClick={() => handleSelect(section)}
              role="button"
              tabIndex="0"
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  handleSelect(section);
                }
              }}
            >
              <div className="section-card-body">
                <h4 className="sec-title">{section.title}</h4>
                <div className="section-badge">
                  {section.len_comments} Comments
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-sections-card">
          <div className="no-sections-content">
            No interactive discussion sections found
          </div>
        </div>
      )}
    </div>
  );
};



export default DiscussionSectionPicker;
