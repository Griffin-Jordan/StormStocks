var selectedTicker;
var svg;

var svg_sent;
 
$(document).ready(function() {
    $('#graphButton').click(function() {
        var ticker = $('#ticker').val(); // Get the selected value from the dropdown menu
        selectedTicker = ticker; // Get the selected value from the dropdown menu
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/api/graph',
            data: JSON.stringify({ 'ticker': ticker }),
            contentType: 'application/json',
            success: function(response) {
                
                //$('#graph').text('LSTM: ' + response.tested);
                     // This is an array
                graphDataLSTM(response.tested)
    
        //if (response && response.tested && response.tested.length > 0) {
            //var predlist = response.tested; // This is an array
            
            // Example: Plotting the graph
            //graphData(predlist); // Call your graphing function with the array data
        //} else {
            //console.error('Empty or invalid data received from the server');
       // }
    },
    error: function(xhr, status, error) {
        console.error(error);
    }
    });
        
    });
    $('#predictButton').click(function() {
        var ticker = $('#ticker').val(); // Assuming you have an input element with ID 'ticker'
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/api/predict',
            data: JSON.stringify({ 'ticker': ticker }), // Send the ticker value as data
            contentType: 'application/json',
            success: function(response) {
                var preturnDict = response.PreturnDict;

                // Access preturn and preturn_sent from the dictionary
                var preturn = preturnDict.preturn;
                var preturn_sent = preturnDict.preturn_sent;

                $('#predict').text('LSTM Predicted Percent Return over the next 60 days: ' + preturn.toFixed(2) + '%');
                $('#predict_sent').text('Sentiment Weighted Monte Carlo Predicted Percent Return over the next 60 days: ' + preturn_sent.toFixed(2) + '%');
            },
            error: function(xhr, status, error) {
                console.error(error);
            }
        });
    });



    $('#sentimentButton').click(function() {
        var ticker = $('#ticker').val();
        selectedTicker = ticker; // Get the selected value from the dropdown menu
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/api/sentiment',
            data: JSON.stringify({ 'ticker': ticker }),
            contentType: 'application/json',
            success: function(response) {
                graphDataSent(response.sentiment_df); 
            
        }
    });
    });


    function graphDataLSTM(JSONdata) {
        // Parse JSON string
        const data = JSON.parse(JSONdata);
        const svgWidth = 800;
        const svgHeight = 400;
        const margin = { top: 50, right: 20, bottom: 50, left: 50 }; //change this to 50
        const width = svgWidth - margin.left - margin.right;
        const height = svgHeight - margin.top - margin.bottom;

        // Extract dates and values for Actual and Forecast
        const actualData = Object.entries(data.Actual).map(([date, value]) => ({ date: new Date(date), value, type: 'Actual' }));
        const forecastData = Object.entries(data.Forecast).map(([date, value]) => ({ date: new Date(date), value, type: 'Forecast' }));

        // Filter out null values from Actual data
        const actualDataFiltered = actualData.filter(d => d.value !== null);

        // Filter out null values from Forecast data
        const forecastDataFiltered = forecastData.filter(d => d.value !== null);

        // Combine filtered data
        const combinedDataFiltered = [...actualDataFiltered, ...forecastDataFiltered];


        d3.select("#graph svg").remove();
        // Append SVG to the element with id "graph"
        var svg = d3.select("#graph")
            .append("svg")
            .attr("width", svgWidth + margin.left + margin.right)
            .attr("height", svgHeight + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // Create scales
        const xScale = d3.scaleTime()
            .domain([d3.min(combinedDataFiltered, d => d.date), d3.max(combinedDataFiltered, d => d.date)])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(combinedDataFiltered, d => d.value)])
            .range([height, 0]);

        // Create scatter plot points for combined data filtered
        svg.selectAll('.dot')
        .data(combinedDataFiltered)
        .enter()
        .append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.date))
        .attr('cy', d => yScale(d.value))
        .attr('r', 5)
        .style('fill', d => d.type === 'Actual' ? 'green' : 'orange') // Color based on type
        .attr('data-type', d => d.type)
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut);

        // Create trend line
        const line = d3.line()
            .x(d => xScale(d.date))
            .y(d => yScale(d.value));

        // Append trend line
        svg.append('path')
            .datum(combinedDataFiltered)
            .attr('class', 'trend-line')
            .attr('fill', 'none')
            .attr('stroke', 'black')
            .attr('stroke-width', 2)
            .attr('d', line);

        // Create axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);

        // Append axes
        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0, ${height})`)
            .call(xAxis);

        svg.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);

        // Add axis labels
        svg.append('text')
            .attr('transform', `translate(${width / 2}, ${height + margin.top + 20})`)
            .style('text-anchor', 'middle')
            .text('Date');

        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text('Stock Price($)');
        
        svg.append("text")
            .attr("x", (width / 2)) 
            .attr("y", 10 - (margin.top / 2)) // Adjust y position for chart title
            .attr("text-anchor", "middle")  
            .style("font-size", "18px") 
            .text("Actual vs. Forecasted " + selectedTicker + " Prices Using LSTM Model");

        // Tooltip
        var tooltip = d3.select("#graph")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("pointer-events", "none")
            .style("background-color", "white")
            .style("border", "1px solid #ccc")
            .style("padding", "10px")
            .style("border-radius", "5px");

        function handleMouseOver(event, d) {
            // Show tooltip on mouseover
            tooltip.html("Type: " + d.type + "<br/>" + "Date: " + d.date.toDateString() + "<br/>" + "Value: " + d.value)
                .style("left", (event.pageX - 50) + "px")
                .style("top", (event.pageY + 20) + "px")
                .transition()
                .duration(200)
                .style("opacity", 0.9);
        }

        function handleMouseOut(event, d) {
            // Hide tooltip on mouseout
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        }
    }



//GRAPH SENTIMENT////



    function graphDataSent(JSONdata) {
        // Parse JSON string
        const data = JSON.parse(JSONdata);
        const svgWidth = 800;
        const svgHeight = 400;
        const margin = { top: 50, right: 20, bottom: 50, left: 50 }; //change this to 50
        const width = svgWidth - margin.left - margin.right;
        const height = svgHeight - margin.top - margin.bottom;

        // Extract dates and values for Actual and Forecast
        const actualData = Object.entries(data.Actual).map(([date, value]) => ({ date: new Date(date), value, type: 'Actual' }));
        const forecastData = Object.entries(data.Forecast).map(([date, value]) => ({ date: new Date(date), value, type: 'Forecast' }));

        // Filter out null values from Actual data
        const actualDataFiltered = actualData.filter(d => d.value !== null);

        // Filter out null values from Forecast data
        const forecastDataFiltered = forecastData.filter(d => d.value !== null);

        // Combine filtered data
        const combinedDataFiltered = [...actualDataFiltered, ...forecastDataFiltered];


        d3.select("#graphsent svg").remove();
        // Append SVG to the element with id "graph"
        var svg_sent = d3.select("#graphsent")
            .append("svg")
            .attr("width", svgWidth + margin.left + margin.right)
            .attr("height", svgHeight + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // Create scales
        const xScale = d3.scaleTime()
            .domain([d3.min(combinedDataFiltered, d => d.date), d3.max(combinedDataFiltered, d => d.date)])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(combinedDataFiltered, d => d.value)])
            .range([height, 0]);

        // Create scatter plot points for combined data filtered
        svg_sent.selectAll('.dot')
        .data(combinedDataFiltered)
        .enter()
        .append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.date))
        .attr('cy', d => yScale(d.value))
        .attr('r', 5)
        .style('fill', d => d.type === 'Actual' ? 'green' : 'orange') // Color based on type
        .attr('data-type', d => d.type)
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut);

        // Create trend line
        const line = d3.line()
            .x(d => xScale(d.date))
            .y(d => yScale(d.value));

        // Append trend line
        svg_sent.append('path')
            .datum(combinedDataFiltered)
            .attr('class', 'trend-line')
            .attr('fill', 'none')
            .attr('stroke', 'black')
            .attr('stroke-width', 2)
            .attr('d', line);

        // Create axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);

        // Append axes
        svg_sent.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0, ${height})`)
            .call(xAxis);

        svg_sent.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);

        // Add axis labels
        svg_sent.append('text')
            .attr('transform', `translate(${width / 2}, ${height + margin.top + 20})`)
            .style('text-anchor', 'middle')
            .text('Date');

        svg_sent.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text('Stock Price($)');
        
        svg_sent.append("text")
            .attr("x", (width / 2)) 
            .attr("y", 10 - (margin.top / 2)) // Adjust y position for chart title
            .attr("text-anchor", "middle")  
            .style("font-size", "18px") 
            .text("Actual vs. Sentiment Weighted Monte Carlo Forecast for " + selectedTicker + " Prices");

        // Tooltip
        var tooltip = d3.select("#graphsent")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("pointer-events", "none")
            .style("background-color", "white")
            .style("border", "1px solid #ccc")
            .style("padding", "10px")
            .style("border-radius", "5px");

        function handleMouseOver(event, d) {
            // Show tooltip on mouseover
            tooltip.html("Type: " + d.type + "<br/>" + "Date: " + d.date.toDateString() + "<br/>" + "Value: " + d.value)
                .style("left", (event.pageX - 50) + "px")
                .style("top", (event.pageY + 20) + "px")
                .transition()
                .duration(200)
                .style("opacity", 0.9);
        }

        function handleMouseOut(event, d) {
            // Hide tooltip on mouseout
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        }
    }




});