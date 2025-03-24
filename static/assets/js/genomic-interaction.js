// Configuration object
const config = {
    itemsPerPage: 10,
    currentPage: 1
};

// Sample data
const geneData = [
    { geneId: 'MsG0180000001.01', chromosome: 'Chr1', startPosition: '146833', endPosition: '150094', strand: '+' },
    { geneId: 'MsG0180000002.01', chromosome: 'Chr1', startPosition: '152749', endPosition: '157114', strand: '+' },
    { geneId: 'MsG0180000003.01', chromosome: 'Chr1', startPosition: '159737', endPosition: '169533', strand: '-' },
    { geneId: 'MsG0180000004.01', chromosome: 'Chr1', startPosition: '171856', endPosition: '175632', strand: '+' },
    { geneId: 'MsG0180000005.01', chromosome: 'Chr1', startPosition: '178924', endPosition: '183651', strand: '-' },
    { geneId: 'MsG0180000006.01', chromosome: 'Chr1', startPosition: '186743', endPosition: '190258', strand: '+' },
    { geneId: 'MsG0180000007.01', chromosome: 'Chr1', startPosition: '193467', endPosition: '197852', strand: '-' },
    { geneId: 'MsG0180000008.01', chromosome: 'Chr1', startPosition: '200164', endPosition: '204589', strand: '+' },
    { geneId: 'MsG0180000009.01', chromosome: 'Chr1', startPosition: '207893', endPosition: '212268', strand: '-' },
    { geneId: 'MsG0180000010.01', chromosome: 'Chr1', startPosition: '215477', endPosition: '219942', strand: '+' },
    { geneId: 'MsG0180000011.01', chromosome: 'Chr1', startPosition: '223156', endPosition: '227631', strand: '-' },
    { geneId: 'MsG0180000012.01', chromosome: 'Chr1', startPosition: '230845', endPosition: '235320', strand: '+' },
    { geneId: 'MsG0180000013.01', chromosome: 'Chr1', startPosition: '238534', endPosition: '243009', strand: '-' },
    { geneId: 'MsG0180000014.01', chromosome: 'Chr1', startPosition: '246223', endPosition: '250698', strand: '+' },
    { geneId: 'MsG0180000015.01', chromosome: 'Chr1', startPosition: '253912', endPosition: '258387', strand: '-' },
    { geneId: 'MsG0180000016.01', chromosome: 'Chr2', startPosition: '146833', endPosition: '150094', strand: '+' },
    { geneId: 'MsG0180000017.01', chromosome: 'Chr2', startPosition: '152749', endPosition: '157114', strand: '-' },
    { geneId: 'MsG0180000018.01', chromosome: 'Chr2', startPosition: '159737', endPosition: '169533', strand: '+' },
    { geneId: 'MsG0180000019.01', chromosome: 'Chr2', startPosition: '171856', endPosition: '175632', strand: '-' },
    { geneId: 'MsG0180000020.01', chromosome: 'Chr2', startPosition: '178924', endPosition: '183651', strand: '+' },
    { geneId: 'MsG0180000021.01', chromosome: 'Chr2', startPosition: '186743', endPosition: '190258', strand: '-' },
    { geneId: 'MsG0180000022.01', chromosome: 'Chr2', startPosition: '193467', endPosition: '197852', strand: '+' },
    { geneId: 'MsG0180000023.01', chromosome: 'Chr2', startPosition: '200164', endPosition: '204589', strand: '-' },
    { geneId: 'MsG0180000024.01', chromosome: 'Chr2', startPosition: '207893', endPosition: '212268', strand: '+' },
    { geneId: 'MsG0180000025.01', chromosome: 'Chr2', startPosition: '215477', endPosition: '219942', strand: '-' },
    { geneId: 'MsG0180000026.01', chromosome: 'Chr3', startPosition: '146833', endPosition: '150094', strand: '+' },
    { geneId: 'MsG0180000027.01', chromosome: 'Chr3', startPosition: '152749', endPosition: '157114', strand: '-' },
    { geneId: 'MsG0180000028.01', chromosome: 'Chr3', startPosition: '159737', endPosition: '169533', strand: '+' },
    { geneId: 'MsG0180000029.01', chromosome: 'Chr3', startPosition: '171856', endPosition: '175632', strand: '-' },
    { geneId: 'MsG0180000030.01', chromosome: 'Chr3', startPosition: '178924', endPosition: '183651', strand: '+' }
];

// Initialize function
function initializeInteractions() {
    // Set up column display/hide functionality
    setupColumnToggle();
    // Set up search functionality
    setupSearch();
    // Initialize table
    renderTable(geneData);
    // Set up pagination
    setupPagination(geneData);
}

// Set up column display/hide
function setupColumnToggle() {
    const checkboxes = document.querySelectorAll('.checkbox-item input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const columnIndex = [...checkboxes].indexOf(this);
            const cells = document.querySelectorAll(`table tr > *:nth-child(${columnIndex + 1})`);
            cells.forEach(cell => {
                cell.style.display = this.checked ? '' : 'none';
            });
        });
    });
}

// Search functionality
function setupSearch() {
    const searchInput = document.querySelector('.search-box input');
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const filteredData = geneData.filter(item => 
            Object.values(item).some(value => 
                value.toLowerCase().includes(searchTerm)
            )
        );
        renderTable(filteredData);
        setupPagination(filteredData);
    });
}

// Render table
function renderTable(data) {
    const tbody = document.querySelector('.genomic-table tbody');
    const start = (config.currentPage - 1) * config.itemsPerPage;
    const end = start + config.itemsPerPage;
    const pageData = data.slice(start, end);

    tbody.innerHTML = pageData.map(item => `
        <tr>
            <td>${item.geneId}</td>
            <td>${item.chromosome}</td>
            <td>${item.startPosition}</td>
            <td>${item.endPosition}</td>
            <td>${item.strand}</td>
        </tr>
    `).join('');

    updatePageInfo(data.length);
}

// Set up pagination
function setupPagination(data) {
    const totalPages = Math.ceil(data.length / config.itemsPerPage);
    const pagination = document.querySelector('.pagination');
    
    let paginationHTML = `
        <button class="page-button" ${config.currentPage === 1 ? 'disabled' : ''} 
                onclick="changePage(${config.currentPage - 1}, ${data.length})">Previous</button>
    `;

    for (let i = 1; i <= totalPages; i++) {
        paginationHTML += `
            <button class="page-button ${config.currentPage === i ? 'active' : ''}"
                    onclick="changePage(${i}, ${data.length})">${i}</button>
        `;
    }

    paginationHTML += `
        <button class="page-button" ${config.currentPage === totalPages ? 'disabled' : ''}
                onclick="changePage(${config.currentPage + 1}, ${data.length})">Next</button>
    `;

    pagination.innerHTML = paginationHTML;
}

// Change page
function changePage(page, totalItems) {
    const totalPages = Math.ceil(totalItems / config.itemsPerPage);
    if (page < 1 || page > totalPages) return;
    config.currentPage = page;
    
    // Get filtered data again
    const searchTerm = document.querySelector('.search-box input').value.toLowerCase();
    const filteredData = geneData.filter(item => 
        Object.values(item).some(value => 
            value.toLowerCase().includes(searchTerm)
        )
    );
    
    renderTable(filteredData);
    setupPagination(filteredData);
}

// Update page information
function updatePageInfo(totalItems) {
    const start = (config.currentPage - 1) * config.itemsPerPage + 1;
    const end = Math.min(config.currentPage * config.itemsPerPage, totalItems);
    const pageInfo = document.querySelector('.page-info');
    pageInfo.textContent = `Showing ${start} to ${end} of ${totalItems} entries`;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initializeInteractions);