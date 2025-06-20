import { Page, Text, View, Document, StyleSheet, Font, Image } from '@react-pdf/renderer';
import { Fragment } from 'react';

export const formatFilterLabel = (filter) => {
    const [key] = filter.split(':');
    if (key.includes(' ')) return key.replace(/^./, c => c.toUpperCase());
    return key.replace(/(?<!^)[A-Z]/g, ' $&').replace(/^./, c => c.toUpperCase());
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
        color: '#050d2d',
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
        color: '#050d2d',
        fontWeight: 'bold',
    },
    value: {
        width: '70%',
        fontSize: 12,
        color: '#050d2d',
    },
    filterItem: {
        margin: 3,
        padding: '3 8',
        backgroundColor: '#050d2d',
        borderRadius: 10,
        fontSize: 10,
        color: 'white',
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
        color: '#050d2d',
        textAlign: 'justify',
    },
    comparison: {
        marginTop: 20,
        padding: 10,
        backgroundColor: '#e3f2fd',
        borderRadius: 5,
        fontSize: 12,
        color: '#050d2d',
        textAlign: 'center',
    },
    metricHighlight: {
        backgroundColor: '#F0F2F5',
        color: 'white',
        padding: 10,
        borderRadius: 5,
        marginTop: 10,
        borderLeft: '4px solid #050d2d',
    },
    metricText: {
        fontSize: 12,
        color: '#050d2d',
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
        color: '#050d2d',
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
        color: '#050d2d',
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
        marginTop: 25,
        marginBottom: 20,
        borderRadius: 8,
        overflow: 'hidden',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        border: '1px solid #E0E0E0',
    },
    tableRow: {
        flexDirection: 'row',
        borderBottomWidth: 1,
        borderBottomColor: '#E8E8E8',
        backgroundColor: '#FFFFFF',
        minHeight: 35,
        alignItems: 'center',
    },
    tableHeader: {
        width: '20%',
        padding: 10,
        backgroundColor: '#050d2d',
        fontSize: 11,
        fontWeight: 'bold',
        color: '#FFFFFF',
        textAlign: 'center',
        borderRight: '1px solid #FFFFFF',
    },
    tableCell: {
        width: '20%',
        padding: 10,
        fontSize: 10,
        textAlign: 'center',
        color: '#050d2d',
        borderRight: '1px solid #F0F0F0',
        backgroundColor: '#FAFAFA',
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
        color: '#050d2d',
    },
    hebrewText: {
        direction: 'rtl',
        textAlign: 'right',
        fontFamily: 'Alef',
    },
    imageCaption: {
        fontSize: 10,
        color: '#050d2d',
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

    const getTopNodesByMetric = (nodes, metric, count = 5) => {
        return [...nodes]
            .sort((a, b) => b[metric] - a[metric])
            .slice(0, count);
    };

    const getNetworkStats = () => {
        if (!research.networkData || !research.networkData.nodes || !research.networkData.links) return null;

        const totalNodes = research.networkData.nodes.length;
        const totalLinks = research.networkData.links.length;
        const avgDegree = totalNodes > 0 ? (totalLinks * 2 / totalNodes).toFixed(2) : 0;
        const totalMessages = research.networkData.nodes.reduce((sum, node) => sum + (node.messages || 0), 0);

        return { totalNodes, totalLinks, avgDegree, totalMessages };
    };

    const networkStats = getNetworkStats();

    return (
        <Document>
            <Page style={styles.page}>
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>Research Report</Text>
                </View>
                {mainImages.length > 0 && (
                    <View style={styles.graphSection}>
                        <Text style={styles.graphTitle}>Network Graph Visualization</Text>
                        {mainImages.map((image, index) =>
                            <Fragment key={index}>
                                <Image
                                    style={styles.graphImage}
                                    src={image.data}
                                    alt="Network Graph"
                                />
                                <Text style={styles.imageCaption}>
                                    (Main Graph) {image.description && ` - ${image.description}`}
                                </Text>
                            </Fragment>

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
                    <View style={styles.row}>
                        <Text style={styles.label}>File:</Text>
                        <Text style={styles.value}>{research.fileName}</Text>
                    </View>
                    <View style={styles.metricHighlight}>
                        <Text style={styles.metricText}>
                            Primary Analysis Metric: {research.metric || 'No specific metric selected'}
                        </Text>
                    </View>
                </View>

                {networkStats !== null && (
                    <View style={styles.section} wrap={false}>
                        <Text style={styles.sectionTitle}>Network Statistics</Text>
                        <View style={styles.row}>
                            <Text style={styles.label}>Total Nodes:</Text>
                            <Text style={styles.value}>{networkStats.totalNodes}</Text>
                        </View>
                        <View style={styles.row}>
                            <Text style={styles.label}>Total Links:</Text>
                            <Text style={styles.value}>{networkStats.totalLinks}</Text>
                        </View>
                        <View style={styles.row}>
                            <Text style={styles.label}>Average Degree:</Text>
                            <Text style={styles.value}>{networkStats.avgDegree}</Text>
                        </View>
                        <View style={styles.row}>
                            <Text style={styles.label}>Total Messages:</Text>
                            <Text style={styles.value}>{networkStats.totalMessages}</Text>
                        </View>
                    </View>
                )}

                {research.networkData?.nodes?.length > 0 && (
                    <View style={styles.section} wrap={false}>
                        <Text style={styles.sectionTitle}>Top Nodes by Centrality Measures</Text>

                        <Text style={[styles.label, { fontSize: 11, marginTop: 10, marginBottom: 5 }]}>
                            Top 5 by Betweenness Centrality:
                        </Text>
                        {getTopNodesByMetric(research.networkData.nodes, 'betweenness').map((node, index) => (
                            <View key={index} style={styles.row}>
                                <Text style={[styles.label, { width: '50%', fontSize: 10 }]}>{node.name}:</Text>
                                <Text style={[styles.value, { width: '50%', fontSize: 10 }]}>{node.betweenness}</Text>
                            </View>
                        ))}

                        <Text style={[styles.label, { fontSize: 11, marginTop: 10, marginBottom: 5 }]}>
                            Top 5 by PageRank:
                        </Text>
                        {getTopNodesByMetric(research.networkData.nodes, 'pagerank').map((node, index) => (
                            <View key={index} style={styles.row}>
                                <Text style={[styles.label, { width: '50%', fontSize: 10 }]}>{node.name}:</Text>
                                <Text style={[styles.value, { width: '50%', fontSize: 10 }]}>{node.pagerank}</Text>
                            </View>
                        ))}

                        <Text style={[styles.label, { fontSize: 11, marginTop: 10, marginBottom: 5 }]}>
                            Top 5 by Message Count:
                        </Text>
                        {getTopNodesByMetric(research.networkData.nodes, 'messages').map((node, index) => (
                            <View key={index} style={styles.row}>
                                <Text style={[styles.label, { width: '50%', fontSize: 10 }]}>{node.name}:</Text>
                                <Text style={[styles.value, { width: '50%', fontSize: 10 }]}>{node.messages} messages</Text>
                            </View>
                        ))}
                    </View>
                )}

                {research.communities.length > 0 && (
                    <View style={styles.section} wrap={false}>
                        <Text style={styles.sectionTitle}>Community Analysis</Text>
                        <View style={styles.row}>
                            <Text style={styles.label}>Total Communities:</Text>
                            <Text style={styles.value}>{research.communities.length}</Text>
                        </View>

                        <View style={styles.comparisonTable} wrap={false}>
                            <View style={[styles.tableRow, { backgroundColor: '#050d2d' }]}>
                                <Text style={styles.tableHeader}>Community</Text>
                                <Text style={styles.tableHeader}>Members</Text>
                                <Text style={styles.tableHeader}>Avg Between</Text>
                                <Text style={styles.tableHeader}>Avg PageRank</Text>
                                <Text style={styles.tableHeader}>Avg Messages</Text>
                            </View>
                            {research.communities.slice(0, 10).map((community, index) => (
                                <View key={index} style={styles.tableRow}>
                                    <Text style={styles.tableCell}>{community.id}</Text>
                                    <Text style={styles.tableCell}>{community.size}</Text>
                                    <Text style={styles.tableCell}>{community.avg_betweenness}</Text>
                                    <Text style={styles.tableCell}>{community.avg_pagerank}</Text>
                                    <Text style={styles.tableCell}>{community.avg_messages}</Text>
                                </View>
                            ))}
                        </View>
                        
                        <Text style={[styles.label, { fontSize: 11, marginTop: 15, marginBottom: 5 }]}>
                            Community Members:
                        </Text>
                        {research.communities.slice(0, 10).map((community, index) => (
                            <View key={index} style={[styles.metricHighlight, { marginTop: 5, padding: 8 }]}>
                                <Text style={[styles.label, { fontSize: 10, marginBottom: 3 }]}>
                                    Community {community.id}:
                                </Text>
                                <Text style={[styles.value, { fontSize: 9 }]}>
                                    {community.nodes?.join(', ') || 'No nodes available'}
                                </Text>
                            </View>
                        ))}
                        
                        {research.communities.length > 10 && (
                            <Text style={[styles.value, { fontSize: 10, marginTop: 5, fontStyle: 'italic' }]}>
                                ... and {research.communities.length - 10} more communities
                            </Text>
                        )}
                    </View>
                )}

                {filters.length > 0 && <View style={styles.section} wrap={false}>
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
                                            <Text style={styles.imageCaption}>(Source Graph) {data.description && `- ${data.description}`}</Text>
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
                            const comparisonFile = research.comparisonFiles[index]
                            return (
                                <Page key={`comparison-${index}`} style={styles.page}>
                                    <Text style={styles.pageNumber}>Comparison Image {index + 1}</Text>

                                    <View style={styles.comparisonSection}>
                                        <Text style={styles.comparisonTitle}>
                                            source: {research?.fileName}. comparison: {comparisonFile}.
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
                                                                <Text style={styles.imageCaption}>
                                                                    (Source Graph) {sourceComparisonImages[0]?.description && `- ${sourceComparisonImages[0]?.description}`}
                                                                </Text>
                                                        </View>
                                                        :
                                                        <>
                                                            <View style={styles.comparisonImageContainer} wrap={false}>
                                                                <Image
                                                                    style={styles.bigSourceImage}
                                                                    src={sourceComparisonImages[0]?.data}
                                                                />
                                                                <Text style={styles.imageCaption}>
                                                                    (Source Graph) {sourceComparisonImages[0]?.description && `- ${sourceComparisonImages[0]?.description}`}
                                                                </Text>
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
                                                    <Text style={styles.imageCaption}>(Comparison Graph) {image.description && `- ${image.description}`}</Text>
                                                </View>
                                            ))}
                                        </View>

                                        <View style={styles.comparisonTable} wrap={false}>
                                            <View style={[styles.tableRow, { backgroundColor: '#F5F5F5' }]}>
                                                <Text style={styles.tableHeader}>Metric</Text>
                                                <Text style={styles.tableHeader}>Original</Text>
                                                <Text style={styles.tableHeader}>Comparison</Text>
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
                                                    {(comparisonStats?.nodeDifference > 0
                                                        ? `+${comparisonStats.nodeDifference}`
                                                        : comparisonStats?.nodeDifference) || ''}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {`${comparisonStats?.nodeChangePercent || ''}${comparisonStats?.nodeChangePercent ? '%' : ''}`}
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
                                                    {(comparisonStats?.linkDifference > 0
                                                        ? `+${comparisonStats.linkDifference}`
                                                        : comparisonStats?.linkDifference) || ''}
                                                </Text>
                                                <Text style={styles.tableCell}>
                                                    {`${comparisonStats?.linkChangePercent || ''}${comparisonStats?.linkChangePercent ? '%' : ''}`}
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

                                        <View style={styles.section} wrap={false}>
                                            <Text style={styles.sectionTitle}>Applied Filters (comparison #{index + 1})</Text>
                                            <View style={styles.filtersContainer}>
                                                {show.comparisonFilters[index] && Object.entries(show.comparisonFilters[index])
                                                    .map(([key, value], filterIndex) => (
                                                        <Text key={filterIndex} style={styles.filterItem}>
                                                            {formatFilterLabel(key) + ': ' + (value || 'false')}
                                                        </Text>
                                                    ))}
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