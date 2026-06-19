class PublicationList {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.loadPublications();
  }

  async loadPublications() {
    if (!this.container) return;

    try {
      const response = await fetch("./publications.json");
      const data = await response.json();
      this.render(data.publications || []);
      window.dispatchEvent(new CustomEvent("publications:rendered"));
    } catch (error) {
      console.error("Failed to load publications:", error);
    }
  }

  render(publications) {
    const fragment = document.createDocumentFragment();

    publications.forEach((publication) => {
      fragment.appendChild(this.createPublication(publication));
    });

    this.container.replaceChildren(fragment);
  }

  createPublication(publication) {
    const article = document.createElement("article");
    article.className = "paper-entry";
    article.id = publication.id;

    article.appendChild(this.createThumbnail(publication));
    article.appendChild(this.createContent(publication));

    return article;
  }

  createThumbnail(publication) {
    const thumbnail = document.createElement("div");
    const image = document.createElement("img");

    thumbnail.className = "paper-thumb";
    image.src = publication.image.src;
    image.alt = publication.image.alt;
    image.className = ["zoomable-image", "paper-image", publication.image.className].filter(Boolean).join(" ");

    thumbnail.appendChild(image);
    return thumbnail;
  }

  createContent(publication) {
    const content = document.createElement("div");

    content.className = "paper-content";
    content.appendChild(this.createTitleLink(publication));
    content.appendChild(this.createAuthors(publication.authors));
    content.appendChild(this.createVenue(publication.venue));
    content.appendChild(this.createLinks(publication.links));
    content.appendChild(this.createDescription(publication.description));

    return content;
  }

  createTitleLink(publication) {
    const link = document.createElement("a");
    const title = document.createElement("span");

    link.href = `#${publication.id}`;
    link.className = "paper-title-link";
    title.className = "papertitle";
    title.textContent = publication.title;

    link.appendChild(title);
    return link;
  }

  createAuthors(authors) {
    const container = document.createElement("div");

    container.className = "paper-authors";

    authors.forEach((author, index) => {
      if (index > 0) container.appendChild(document.createTextNode(", "));
      container.appendChild(this.createAuthor(author));
      if (author.suffix) container.appendChild(document.createTextNode(author.suffix));
    });

    return container;
  }

  createAuthor(author) {
    if (author.collaborator) {
      const collaborator = document.createElement("span");
      collaborator.dataset.collaborator = author.collaborator;
      return collaborator;
    }

    if (author.strong) {
      const strong = document.createElement("strong");
      strong.textContent = author.name;
      return strong;
    }

    return document.createTextNode(author.name);
  }

  createVenue(venue) {
    const container = document.createElement("div");
    const venueElement = document.createElement(venue.tag || "span");

    container.className = "paper-venue";
    venueElement.textContent = venue.text;
    container.appendChild(venueElement);

    return container;
  }

  createLinks(links) {
    const container = document.createElement("div");

    container.className = "paper-links";

    links.forEach((linkData, index) => {
      const link = document.createElement("a");

      if (index > 0) container.appendChild(document.createTextNode(" / "));
      link.href = linkData.url;
      link.textContent = linkData.label;
      container.appendChild(link);
    });

    return container;
  }

  createDescription(description) {
    const paragraph = document.createElement("p");

    paragraph.className = "paper-description";
    paragraph.textContent = description;

    return paragraph;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new PublicationList("publication-list");
});
