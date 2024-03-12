// Load predicted price data from the backend
// Replace 'http://localhost:5000/api/predicted_prices' with the actual endpoint
// Make sure the endpoint returns data in the format { date: string, predicted_price: number }

d3.json('http://localhost:5000/api/predict')
  .then(function(data) {
    // Parse dates and convert predicted_price to number
    data.forEach(function(d) {
      d.date = new Date(d.date);
      d.predicted_price = +d.predicted_price;
    });

    // Set up the dimensions and margins for the chart
    var margin = { top: 20, right: 30, bottom: 30, left: 50 },
        width = 600 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    // Create SVG element
    var svg = d3.select('#chart')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Set up scales
    var x = d3.scaleTime()
              .domain(d3.extent(data, function(d) { return d.date; }))
              .range([0, width]);

    var y = d3.scaleLinear()
              .domain([0, d3.max(data, function(d) { return d.predicted_price; })])
              .nice()
              .range([height, 0]);

    // Create line
    var line = d3.line()
                 .x(function(d) { return x(d.date); })
                 .y(function(d) { return y(d.predicted_price); });

    // Add line to SVG
    svg.append('path')
       .datum(data)
       .attr('class', 'line')
       .attr('d', line);

    // Add dots to the line for each data point
    svg.selectAll('.dot')
       .data(data)
       .enter().append('circle')
       .attr('class', 'dot')
       .attr('cx', function(d) { return x(d.date); })
       .attr('cy', function(d) { return y(d.predicted_price); })
       .attr('r', 4)
       .on('mouseover', function(d) {
          // Show predicted price on hover
          tooltip.transition()
                 .duration(200)
                 .style('opacity', .9);
          tooltip.html('Predicted Price: ' + d.predicted_price)
                 .style('left', (d3.event.pageX + 5) + 'px')
                 .style('top', (d3.event.pageY - 28) + 'px');
       })
       .on('mouseout', function(d) {
          // Hide tooltip on mouseout
          tooltip.transition()
                 .duration(500)
                 .style('opacity', 0);
       });

    // Add X axis
    svg.append('g')
       .attr('transform', 'translate(0,' + height + ')')
       .call(d3.axisBottom(x));

    // Add Y axis
    svg.append('g')
       .call(d3.axisLeft(y));

    // Add tooltip
    var tooltip = d3.select('#chart')
                    .append('div')
                    .attr('class', 'tooltip')
                    .style('opacity', 0);
  })
  .catch(function(error) {
    console.log('Error fetching data:', error);
  });