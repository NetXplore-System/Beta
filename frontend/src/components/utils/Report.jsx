import { Page, Text, View, Document, StyleSheet, Font, Image } from '@react-pdf/renderer';

export const formatFilterLabel = (filter) => {
    const [key] = filter.split(':');
    return key
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalize each word
        .join(' ');
};

Font.register({
    family: 'Alef',
    src: '/Alef/Alef-Regular.ttf',
});

const styles = StyleSheet.create({
    page: {
        padding: 40,
        backgroundColor: '#FCFCFC',
        fontFamily: 'Alef',
    },
    header: {
        marginBottom: 20,
        borderBottom: '1px solid #050d2d',
        paddingBottom: 10,
    },
    headerTitle: {
        fontSize: 24,
        color: '#050d2d',
        textAlign: 'center',
        fontWeight: 'bold',
    },
    section: {
        marginBottom: 20,
        padding: 10,
        backgroundColor: '#F8F9FA',
        borderRadius: 5,
    },
    sectionTitle: {
        fontSize: 16,
        marginBottom: 10,
        color: '#158582',
        fontWeight: 'bold',
        borderBottom: '1px solid #DDD',
        paddingBottom: 5,
    },
    row: {
        flexDirection: 'row',
        marginBottom: 5,
    },
    label: {
        width: '30%',
        fontSize: 12,
        color: '#555',
        fontWeight: 'bold',
    },
    value: {
        width: '70%',
        fontSize: 12,
        color: '#333',
    },
    filterItem: {
        margin: 3,
        padding: '3 8',
        backgroundColor: '#e0f2f1',
        borderRadius: 10,
        fontSize: 10,
        color: '#00796b',
        display: 'inline-block',
    },
    filtersContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginTop: 5,
    },
    conclusion: {
        fontSize: 12,
        lineHeight: 1.6,
        color: '#333',
        textAlign: 'justify',
    },
    comparison: {
        marginTop: 20,
        padding: 10,
        backgroundColor: '#e3f2fd',
        borderRadius: 5,
        fontSize: 12,
        color: '#1565c0',
        textAlign: 'center',
    },
    metricHighlight: {
        backgroundColor: '#f0f7f6',
        padding: 10,
        borderRadius: 5,
        marginTop: 10,
        borderLeft: '4px solid #158582',
    },
    metricText: {
        fontSize: 12,
        color: '#2a625e',
        fontWeight: 'semibold',
    },
    graphSection: {
        marginVertical: 15,
        padding: 10,
        backgroundColor: '#FFF',
        borderRadius: 8,
        boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
    },
    graphTitle: {
        fontSize: 14,
        color: '#158582',
        marginBottom: 8,
        fontWeight: 'semibold',
    },
    comparisonSection: {
        marginTop: 15,
        paddingTop: 15,
        borderTop: '1px solid #DDD',
    },
    comparisonTitle: {
        fontSize: 14,
        color: '#1565c0',
        fontWeight: 'bold',
        marginBottom: 5,
        textAlign: 'center',
    },
    graphImage: {
        width: '100%',
        height: 210,
        borderRadius: 4,
        marginTop: 10,
        border: '1px solid #EEE',
    },
    comparisonImage: {
        width: '75%', 
        height: 300,
        borderRadius: 4,
        border: '1px solid #EEE',
    },
    bigSourceImage: {
        width: '100%',
        height: 250,
        borderRadius: 4,
        border: '1px solid #EEE',
    },
    comparisonImagesContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap', 
        justifyContent: 'center', 
        // marginVertical: 10,
    },
    comparisonImageContainer: {
        flexBasis: '100%',
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        marginBottom: 10, 
    },
    comparisonTable: {
        width: '100%',
        marginTop: 115,
        marginBottom: 15,
    },
    tableRow: {
        flexDirection: 'row',
        borderBottomWidth: 1,
        borderBottomColor: '#DDD',
        backgroundColor: '#FFF',
    },
    tableHeader: {
        width: '20%',
        padding: 8,
        backgroundColor: '#F5F5F5',
        fontSize: 10,
        fontWeight: 'bold',
        color: '#333',
        textAlign: 'center',
        borderBottom: 1,
        borderBottomColor: '#DDD',
    },
    tableCell: {
        width: '20%',
        padding: 8,
        fontSize: 10,
        textAlign: 'center',
        color: '#444',
    },
    footer: {
        position: 'absolute',
        bottom: 30,
        left: 30,
        right: 30,
        textAlign: 'center',
        color: '#888',
        fontSize: 10,
        borderTop: '1px solid #DDD',
        paddingTop: 10,
    },
    dateText: {
        position: 'absolute',
        top: 30,
        right: 30,
        fontSize: 10,
        color: '#666',
    },
    hebrewText: {
        direction: 'rtl',
        textAlign: 'right',
        fontFamily: 'Alef',
    },
    imageCaption: {
        fontSize: 10,
        color: '#666',
        textAlign: 'center',
        marginTop: 5,
    },
});

const Report = ({ research, show }) => {
    const mainImages = show.images.filter((image) => image.type === "main" && image.selected);
    const sourceComparisonImages = show.images.filter((image) => image.type === "source" && image.selected);
    let comparisonImages = show.images.filter((image) => image.type === "comparison" && image.selected);
    const groupedComparisonImages = comparisonImages.reduce((acc, image) => {
        const index = image.index;
        if (!acc[index]) {
            acc[index] = [];
        }
        acc[index].push(image);
        return acc;
    }, {});

    const groupedComparisonArray = Object.entries(groupedComparisonImages).map(([_, images]) => ([...images]));
    groupedComparisonArray.sort((a, b) => a.index - b.index);
    
    const filters = show.filters.filter((filter) => filter.selected).map((filter) => research.filters[filter.index]);

    return (
        <Document>
            <Page style={styles.page}>
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>Research Report</Text>
                </View>
                {mainImages.length && (
                    <View style={styles.graphSection}>
                        <Text style={styles.graphTitle}>Network Graph Visualization</Text>
                        {mainImages.map((image) =>
                            <>
                                <Image
                                    style={styles.graphImage}
                                    src={image.data}
                                    alt="Network Graph"
                                />
                                <Text style={styles.imageCaption}>
                                    {image.description && ` - ${image.description}`}
                                </Text>
                            </>

                        )}
                    </View>
                )}
                <View style={styles.section} wrap={false}>
                    <Text style={styles.sectionTitle}>Research Information</Text>
                    <View style={styles.row}>
                        <Text style={styles.label}>Research Name:</Text>
                        <Text style={styles.value}>{research.name}</Text>
                    </View>
                    <View style={styles.row}>
                        <Text style={styles.label}>Researcher:</Text>
                        <Text style={styles.value}>{research.researcherName}</Text>
                    </View>
                    <View style={styles.metricHighlight}>
                        <Text style={styles.metricText}>
                            Primary Analysis Metric: {research.metric || 'No specific metric selected'}
                        </Text>
                    </View>
                </View>

                {filters && <View style={styles.section} wrap={false}>
                    <Text style={styles.sectionTitle}>Applied Filters</Text>
                    <View style={styles.filtersContainer}>
                        {filters.map((filter, index) => (
                            <Text key={index} style={styles.filterItem}>
                                {`${formatFilterLabel(filter)}: ${filter.split(':')[1].trim()}`}
                            </Text>
                        ))}
                    </View>
                </View>}

                <View style={styles.section} wrap={false}>
                    <Text style={styles.sectionTitle}>Research Conclusion</Text>
                    <Text style={styles.conclusion}>{research.conclusion || research.review}</Text>
                </View>

                {research.hasComparison && (
                    <View style={styles.comparison}>
                        <Text>This research includes {research.stats.length} comparison studies.</Text>
                    </View>
                )}
            </Page>

            {research.hasComparison ? (
                <>
                    {sourceComparisonImages.length > 1 && (
                        <Page key="source-comparison" style={styles.page}>
                            <Text style={styles.pageNumber}>Source Comparison Images</Text>
                            <View style={styles.comparisonSection}>
                                <View style={styles.comparisonImagesContainer}>
                                    {sourceComparisonImages.map((data, index) => (
                                        <View key={index} style={styles.comparisonImageContainer} wrap={false}>
                                            {data.width === "big" ?
                                                <Image
                                                    style={[styles.bigSourceImage, { marginTop: 20 }]}
                                                    src={data.data}
                                                />
                                                :
                                                <Image
                                                    style={styles.comparisonImage}
                                                    src={data.data}
                                                />
                                            }
                                            {data.description && (
                                                <Text style={styles.imageCaption}>{data.description}</Text>
                                            )}
                                        </View>
                                    ))}
                                </View>
                            </View>
                        </Page>
                    )}

                    {comparisonImages.length > 0 &&
                        groupedComparisonArray.map((data, index) => {
                            const comparisonStats = research.stats.find(
                                (state) => state.index === data[0].index
                            );
                            return (
                                <Page key={`comparison-${index}`} style={styles.page}>
                                    <Text style={styles.pageNumber}>Comparison Image</Text>

                                    <View style={styles.comparisonSection}>
                                        <Text style={styles.comparisonTitle}>
                                            Comparison #{index + 1} Page: {comparisonStats?.fileName}
                                        </Text>

                                        <View style={styles.comparisonImagesContainer}>
                                            {sourceComparisonImages.length === 1 &&
                                                <>
                                                    {sourceComparisonImages[0].width === "small" ?
                                                        <View style={styles.comparisonImageContainer} wrap={false}>
                                                            <Image
                                                                style={styles.comparisonImage}
                                                                src={sourceComparisonImages[0]?.data}
                                                            />
                                                            {sourceComparisonImages[0]?.description && (
                                                                <Text style={styles.imageCaption}>
                                                                    {sourceComparisonImages[0]?.description}
                                                                </Text>
                                                            )}
                                                        </View>
                                                        :
                                                        <>
                                                            <View style={styles.comparisonImageContainer} wrap={false}>
                                                                <Image
                                                                    style={styles.bigSourceImage}
                                                                    src={sourceComparisonImages[0]?.data}
                                                                />
                                                                {sourceComparisonImages[0]?.description && (
                                                                    <Text style={styles.imageCaption}>
                                                                        {sourceComparisonImages[0]?.description}
                                                                    </Text>
                                                                )}
                                                            </View>
                                                        </>}
                                                </>
                                            }
                                            {data.map((image, index) => (
                                                <View key={index} style={styles.comparisonImageContainer} wrap={false}>
                                                    <Image
                                                        style={styles.comparisonImage}
                                                        src={image.data}
                                                    />
                                                    {image.description && (
                                                        <Text style={styles.imageCaption}>{image.description}</Text>
                                                    )}
                                                </View>
                                            ))}
                                        </View>

                                        <View style={styles.comparisonTable} wrap={false}>
                                            <View style={[styles.tableRow, { backgroundColor: '#F5F5F5' }]}>
                                                <Text style={styles.tableHeader}>Metric</Text>
                                                <Text style={styles.tableHeader}>Original Network</Text>
                                                <Text style={styles.tableHeader}>Comparison Network</Text>
                                                <Text style={styles.tableHeader}>Difference</Text>
                                                <Text style={styles.tableHeader}>Change %</Text>
                                            </View>

                                            <View style={styles.tableRow}>
                                                <Text style={styles.tableCell}>Node Count</Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.originalNodeCount}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.comparisonNodeCount}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.nodeDifference > 0
                                                        ? `+${comparisonStats.nodeDifference}`
                                                        : comparisonStats?.nodeDifference}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.nodeChangePercent}%
                                                </Text>
                                            </View>

                                            <View style={styles.tableRow}>
                                                <Text style={styles.tableCell}>Edge Count</Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.originalLinkCount}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.comparisonLinkCount}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.linkDifference > 0
                                                        ? `+${comparisonStats.linkDifference}`
                                                        : comparisonStats?.linkDifference}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {comparisonStats?.linkChangePercent}%
                                                </Text>
                                            </View>

                                            <View style={styles.tableRow}>
                                                <Text style={styles.tableCell}>Common Nodes</Text>
                                                <Text style={[styles.tableCell, { width: '40%' }]}>
                                                    {comparisonStats?.commonNodesCount}
                                                </Text>
                                                <Text style={[styles.tableCell, { width: '40%' }]}>
                                                    {(
                                                        (comparisonStats?.commonNodesCount /
                                                            comparisonStats?.originalNodeCount) *
                                                        100
                                                    ).toFixed(2)}
                                                    % of original network
                                                </Text>
                                            </View>
                                        </View>
                                    </View>
                                    <View style={styles.footer} fixed>
                                        <Text>
                                            Generated on {new Date().toLocaleDateString()} - Network Analysis
                                            Platform
                                        </Text>
                                    </View>
                                </Page>
                            );
                        })}
                </>
            ) : null}
        </Document>
    )
};

export default Report;