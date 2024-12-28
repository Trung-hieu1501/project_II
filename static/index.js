// Lấy các phần tử HTML cần thao tác
const input = document.getElementById('stock_code');
const suggestionsList = document.getElementById('suggestions');

// Khi người dùng nhập vào ô tìm kiếm
input.addEventListener('input', async () => {
    const query = input.value.trim(); // Lấy giá trị từ ô tìm kiếm

    if (query) {
        try {
            // Fetch dữ liệu gợi ý (ví dụ dùng API)
            const response = await fetch(`/suggest?query=${query}`);
            const suggestions = await response.json(); // Mảng các mã chứng khoán

            if (suggestions.length > 0) {
                // Render danh sách gợi ý, bao gồm cả tên công ty
                suggestionsList.innerHTML = suggestions
                    .map(suggestion => {
                        return `
                            <li onclick="selectSuggestion('${suggestion.symbol}')">
                                ${suggestion.symbol} - ${suggestion.company_name}
                            </li>
                        `;
                    })
                    .join('');
                suggestionsList.style.display = 'block'; // Hiển thị danh sách
                suggestionsList.style.width = `${input.offsetWidth}px`; // Khớp chiều rộng
            } else {
                suggestionsList.style.display = 'none'; // Ẩn nếu không có gợi ý
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            suggestionsList.style.display = 'none';
        }
    } else {
        suggestionsList.style.display = 'none'; // Ẩn nếu không có nội dung nhập
    }
});

// Khi người dùng chọn một gợi ý
function selectSuggestion(symbol) {
    input.value = symbol; // Điền mã được chọn vào ô input
    suggestionsList.style.display = 'none'; // Ẩn danh sách gợi ý
}

// Ẩn danh sách khi click ra ngoài vùng tìm kiếm
document.addEventListener('click', (event) => {
    if (!event.target.closest('.track-stock')) {
        suggestionsList.style.display = 'none';
    }
});


// Tối ưu danh mục đầu tư


