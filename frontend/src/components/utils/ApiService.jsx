import { toast } from 'react-hot-toast';

export const BASE_URL = import.meta.env.VITE_API_URL;


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

    return await response.json();
  } catch (error) {
    console.error("Error uploading file:", error);
    throw new Error("An error occurred during the upload.");
  }
};

export const deleteFile = async (filename) => {
  try {
    const response = await fetch(`${BASE_URL}/delete/${filename}`, {
      method: "DELETE",
    });
    return await response.json();
  } catch (error) {
    console.error("Error deleting file:", error);
    throw new Error("An error occurred during the delete operation.");
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
    return await response.json();
  } catch (error) {
    console.error("Error saving form:", error);
    throw new Error("An error occurred while saving the form.");
  }
};

export const analyzeNetwork = async (filename, params) => {
  try {
    const url = `${BASE_URL}/analyze/network/${filename}?${params.toString()}`;
    console.log("Request URL:", url);

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    console.error("Error during network analysis:", error);
    throw new Error("An error occurred during network analysis.");
  }
};

export const detectCommunities = async (filename, params) => {
  try {
    params.append("algorithm", "louvain");
    const url = `${BASE_URL}/analyze/communities/${filename}?${params.toString()}`;
    console.log("Community detection URL:", url);

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    console.error("Error during community detection:", error);
    throw new Error("An error occurred during community detection.");
  }
};

export const compareNetworks = async (params) => {
  try {
    const url = `${BASE_URL}/analyze/compare-networks?${params.toString()}`;
    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    console.error("Error during network comparisons:", error);
    throw new Error("An error occurred during network comparisons");
  }
};
