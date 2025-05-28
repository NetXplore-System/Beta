import { toast } from "sonner";
import { logoutUser } from "../../redux/user/userSlice"; 
import store from "../../redux/store";
export const BASE_URL = import.meta.env.VITE_API_URL;

export const fetchWithAuth = async (url, options = {}) => {
  const token = localStorage.getItem("token");

  const headers = {
    ...(options.headers || {}),
    Authorization: `Bearer ${token}`,
  };

  const response = await fetch(url, { ...options, headers });

  if (response.status === 401 || response.status === 403) {
    toast.error("Session expired. Please log in again.");
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    store.dispatch(logoutUser());
    window.location.href = "/#/signin"; 
    throw new Error("Session expired");
  }

  return response;
};


export const uploadFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BASE_URL}/upload`, {
      method: "POST",
      body: formData,
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error("Error uploading file:", error);
    throw Error(error || "An error occurred during the upload.");
  }
};

export const deleteFile = async (filename) => {
  try {
    const response = await fetch(`${BASE_URL}/delete/${filename}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error(response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error("Error deleting file:", error);
    throw new Error(error || "An error occurred during the delete operation.");
  }
};

export const saveFormToDB = async (formData) => {
  try {
    const response = await fetch(`${BASE_URL}/save-form`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });
    if (!response.ok) {
      throw new Error(response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error("Error saving form:", error);
    throw new Error(error || "An error occurred while saving the form.");
  }
};

export const analyzeNetwork = async (filename, params) => {
  try {
    const url = `${BASE_URL}/analyze/network/${filename}?${params.toString()}`;
    console.log("Request URL:", url);

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(response.statusText);
    }
    return await response.json();
  } catch (error) {
    console.error("Error during network analysis:", error);
    throw new Error(error || "An error occurred during network analysis.");
  }
};

export const analyzeDecayingNetwork = async (filename, params) => {
  try {
    const url = `${BASE_URL}/analyze/decaying-network/${filename}?${params.toString()}`;
    console.log("Request URL:", url);

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error("Error during decaying network analysis:", error);
    throw new Error(error || "An error occurred during decaying network analysis.");
  }
};

export const detectCommunities = async (filename, params) => {
  try {
    const url = `${BASE_URL}/analyze/communities/${filename}?${params.toString()}`;
    console.log("Community detection URL:", url);

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error("Error during community detection:", error);
    throw new Error(error || "An error occurred during community detection.");
  }
};



export const compareNetworks = async (params) => {
  try {
    const url = `${BASE_URL}/analyze/compare-networks?${params.toString()}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    return await response.json();
  } catch (error) {
    console.error("Error during network comparisons:", error);
    throw new Error("An error occurred during network comparisons");
  }
};

export const fetchWikipediaData = async (url) => {
  try {
    const response = await fetch(`${BASE_URL}/fetch-wikipedia-data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    if (data.nodes && data.links && data.content) {
      return {
        nodes: data.nodes.map((node) => ({ ...node, id: String(node.id) })),
        links: data.links.map((link) => ({
          ...link,
          source: String(link.source),
          target: String(link.target),
        })),
        content: data.content,
        opinions: data.opinions || { for: 0, against: 0, neutral: 0 },
      };
    } else {
      throw new Error("No valid discussion data found on this Wikipedia page.");
    }
  } catch (error) {
    console.error("Error loading Wikipedia data:", error);
    throw new Error(error.message || "Failed to fetch Wikipedia data.");
  }
};

export const saveToDB = async (
  id,
  token,
  name,
  description,
  file,
  params,
  selectedMetric = "Degree Centrality",
  comparison,
  platform = "whatsapp",
  communities = []
) => {

  try {
    const formData = new FormData();

    formData.append("research_name", name);
    formData.append("researcher_id", id);
    formData.append("description", description);
    comparison.hasComparison && formData.append("comparison", JSON.stringify(comparison.data));
    formData.append("file_name", file);
    formData.append("selected_metric", selectedMetric);
    formData.append("platform", platform);
    formData.append("communities", JSON.stringify(communities || []));


    const response = await fetch(
     `${BASE_URL}/save-research?${params.toString()}`,
      {
        method: "POST",
        body: formData,
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    );
    if (!response.ok) {
      const { detail } = await response.json();
      console.error("Error response:", detail);
      throw new Error(detail || "An error occurred while saving the research.");
    }

    return "Research saved successfully!";

  } catch (error) {
    console.error("Save error:", error);
    throw new Error("An error occurred while saving the research.");
  }
};


export const deleteResearch = async (researchId, token) => {
  try {
    const response = await fetch(`${BASE_URL}/research/${researchId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    if (!response.ok) {
      throw Error('Failed to delete research');
    }
    
    return "Research deleted successfully";
  } catch (error) {
    console.error('Error deleting research:', error);
    throw Error('Error deleting research');
  }
}

export const detectWikipediaCommunities = async (filename, params) => {
  try {
    const url = `${BASE_URL}/analyze/wikipedia-communities/${filename}?${params.toString()}`;
    console.log("Wikipedia community detection URL:", url);

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return await response.json();
  } catch (error) {
    console.error("Error during Wikipedia community detection:", error);
    throw new Error("An error occurred during Wikipedia community detection.");
  }
};
